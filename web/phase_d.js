const BATCH = 2;
const TYPE12_POINT_COUNT = 7;
const TYPE3_SOURCE_COUNT = 7;
const TYPE3_TARGET_COUNT = 6;
const SIGMA = 2.0;
const F32_EPS = 1e-5;
const DF64_EPS = 1e-8;
const qs = new URLSearchParams(location.search);
const statusEl = document.getElementById("status");
const runButton = document.getElementById("run");

function splitDf64(value) {
  const hi = Math.fround(value);
  const lo = Math.fround(value - hi);
  return [hi, lo, Number(hi) + Number(lo)];
}

function packScalars(values, precisionName) {
  if (precisionName === "F32") {
    const canonical = values.map(Math.fround);
    return { upload: new Float32Array(canonical), canonical };
  }
  const canonical = values.map((value) => splitDf64(value)[2]);
  return { upload: new Float64Array(values), canonical };
}

function packComplex(values, precisionName) {
  return packScalars(values, precisionName);
}

function asBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error("download returned an unsupported byte container");
}

function decodeComplex(bytesValue, precisionName) {
  const bytes = asBytes(bytesValue);
  if (bytes.byteLength % 4 !== 0) throw new Error("download is not f32-word aligned");
  const words = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const stride = precisionName === "F32" ? 2 : 4;
  if (words.length % stride !== 0) throw new Error("download has an incomplete complex value");
  const values = [];
  for (let i = 0; i < words.length; i += stride) {
    values.push(
      precisionName === "F32"
        ? [Number(words[i]), Number(words[i + 1])]
        : [Number(words[i]) + Number(words[i + 1]), Number(words[i + 2]) + Number(words[i + 3])],
    );
  }
  return values;
}

function uploadPacked(context, packed, precisionName) {
  return precisionName === "F32"
    ? context.upload(packed.upload)
    : context.uploadDf64(packed.upload);
}

function complexPairs(scalars) {
  if (scalars.length % 2 !== 0) throw new Error("complex scalar list has odd length");
  const values = [];
  for (let i = 0; i < scalars.length; i += 2) values.push([scalars[i], scalars[i + 1]]);
  return values;
}

function rotate(value, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [value[0] * cosine - value[1] * sine, value[0] * sine + value[1] * cosine];
}

function modeVector(linear, shape) {
  return shape.map((length) => {
    const index = linear % length;
    linear = Math.floor(linear / length);
    return index - Math.floor(length / 2);
  });
}

function pointPhase(points, point, dimensions, vector) {
  let phase = 0;
  for (let axis = 0; axis < dimensions; axis += 1) {
    phase += points[point * dimensions + axis] * vector[axis];
  }
  return phase;
}

function referenceType1(shape, points, strengths, isign, batch) {
  const dimensions = shape.length;
  const pointCount = points.length / dimensions;
  const modeCount = shape.reduce((product, value) => product * value, 1);
  const strengthPairs = complexPairs(strengths);
  const output = [];
  for (let transform = 0; transform < batch; transform += 1) {
    for (let linear = 0; linear < modeCount; linear += 1) {
      const modes = modeVector(linear, shape);
      let re = 0;
      let im = 0;
      for (let point = 0; point < pointCount; point += 1) {
        const rotated = rotate(
          strengthPairs[transform * pointCount + point],
          isign * pointPhase(points, point, dimensions, modes),
        );
        re += rotated[0];
        im += rotated[1];
      }
      output.push([re, im]);
    }
  }
  return output;
}

function referenceType2(shape, points, coefficients, isign, batch) {
  const dimensions = shape.length;
  const pointCount = points.length / dimensions;
  const modeCount = shape.reduce((product, value) => product * value, 1);
  const coefficientPairs = complexPairs(coefficients);
  const output = [];
  for (let transform = 0; transform < batch; transform += 1) {
    for (let point = 0; point < pointCount; point += 1) {
      let re = 0;
      let im = 0;
      for (let linear = 0; linear < modeCount; linear += 1) {
        const rotated = rotate(
          coefficientPairs[transform * modeCount + linear],
          isign * pointPhase(points, point, dimensions, modeVector(linear, shape)),
        );
        re += rotated[0];
        im += rotated[1];
      }
      output.push([re, im]);
    }
  }
  return output;
}

function referenceType3(source, target, strengths, dimensions, isign, batch) {
  const sourceCount = source.length / dimensions;
  const targetCount = target.length / dimensions;
  const strengthPairs = complexPairs(strengths);
  const output = [];
  for (let transform = 0; transform < batch; transform += 1) {
    for (let targetIndex = 0; targetIndex < targetCount; targetIndex += 1) {
      let re = 0;
      let im = 0;
      const frequency = target.slice(targetIndex * dimensions, (targetIndex + 1) * dimensions);
      for (let sourceIndex = 0; sourceIndex < sourceCount; sourceIndex += 1) {
        const rotated = rotate(
          strengthPairs[transform * sourceCount + sourceIndex],
          isign * pointPhase(source, sourceIndex, dimensions, frequency),
        );
        re += rotated[0];
        im += rotated[1];
      }
      output.push([re, im]);
    }
  }
  return output;
}

function relativeL2(actual, expected) {
  if (actual.length !== expected.length) {
    throw new Error(`output length ${actual.length}, expected ${expected.length}`);
  }
  let error2 = 0;
  let reference2 = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const dre = actual[i][0] - expected[i][0];
    const dim = actual[i][1] - expected[i][1];
    error2 += dre * dre + dim * dim;
    reference2 += expected[i][0] ** 2 + expected[i][1] ** 2;
  }
  return Math.sqrt(error2) / Math.max(Math.sqrt(reference2), Number.MIN_VALUE);
}

function tolerance(kind, dimensions, precisionName) {
  if (precisionName === "Df64") return 24 * DF64_EPS;
  if (kind === "type-3") return 100 * F32_EPS;
  return [0, 4, 20, 32][dimensions] * F32_EPS;
}

function type12Points(dimensions) {
  const points = [];
  for (let point = 0; point < TYPE12_POINT_COUNT; point += 1) {
    for (let axis = 0; axis < dimensions; axis += 1) {
      let value;
      if (point === 0) value = -Math.PI + (axis + 1) * 1e-6;
      else if (point === 1) value = Math.PI - (axis + 1) * 1e-6;
      else if (point === 2 || point === 3) value = 0.375 - 0.125 * axis;
      else if (point === 4) value = -0.2 + axis * 0.07;
      else if (point === 5) value = -2.4 + axis * 0.11;
      else value = 2.2 - axis * 0.09;
      points.push(value);
    }
  }
  return points;
}

function type3Bounds(dimensions) {
  const sourceCenters = [0.375, -0.5, 0.75];
  const sourceHalfWidths = [1.0, 1.25, 0.75];
  const targetCenters = [0.75, -1.0, 0.5];
  const targetHalfWidths = [2.25, 1.75, 2.5];
  const source = [];
  const target = [];
  for (let axis = 0; axis < dimensions; axis += 1) {
    source.push(sourceCenters[axis] - sourceHalfWidths[axis], sourceCenters[axis] + sourceHalfWidths[axis]);
    target.push(targetCenters[axis] - targetHalfWidths[axis], targetCenters[axis] + targetHalfWidths[axis]);
  }
  return { source, target };
}

function intervalPoints(bounds, count) {
  const dimensions = bounds.length / 2;
  const points = [];
  for (let point = 0; point < count; point += 1) {
    for (let axis = 0; axis < dimensions; axis += 1) {
      const lower = bounds[2 * axis];
      const upper = bounds[2 * axis + 1];
      const center = 0.5 * (lower + upper);
      const halfWidth = 0.5 * (upper - lower);
      let value;
      if (point === 0) value = lower;
      else if (point === 1) value = upper;
      else if (point === 2 || point === 3) value = center;
      else if (point === 4) value = center + (axis + 1) * 1e-5;
      else if (point === 5) value = center - 0.47 * halfWidth;
      else value = center + 0.61 * halfWidth;
      points.push(value);
    }
  }
  return points;
}

function complexValues(batch, count, bias) {
  const values = [];
  for (let transform = 0; transform < batch; transform += 1) {
    for (let index = 0; index < count; index += 1) {
      const x = index + 1;
      const offset = bias + transform * 0.117;
      values.push(Math.sin(x * 0.31 + offset) * 0.7, Math.cos(x * 0.23 - offset) * 0.5);
    }
  }
  return values;
}

async function runType12(context, module, shape, isign, precisionName, results) {
  const precision = module.WebFftPrecision[precisionName];
  const eps = precisionName === "F32" ? F32_EPS : DF64_EPS;
  const modeCount = shape.reduce((product, value) => product * value, 1);
  const points = packScalars(type12Points(shape.length), precisionName);
  const strengths = packComplex(complexValues(BATCH, TYPE12_POINT_COUNT, 0.37), precisionName);
  const coefficients = packComplex(complexValues(BATCH, modeCount, -0.29), precisionName);
  const gpuPoints = uploadPacked(context, points, precisionName);

  const type1 = await context.createNufftType1Plan(
    new Uint32Array(shape), TYPE12_POINT_COUNT, BATCH, eps, isign,
    module.WebNufftModeOrder.Centered, SIGMA, precision,
  );
  const gpuStrengths = uploadPacked(context, strengths, precisionName);
  const type1Output = context.createBuffer(type1.outputBytes);
  await type1.execute(gpuPoints, gpuStrengths, type1Output, BATCH);
  const type1Actual = decodeComplex(await context.download(type1Output), precisionName);
  const type1Expected = referenceType1(shape, points.canonical, strengths.canonical, isign, BATCH);
  recordAccuracy(results, "type-1", shape.length, precisionName, isign, type1Actual, type1Expected);

  const type2 = await context.createNufftType2Plan(
    new Uint32Array(shape), TYPE12_POINT_COUNT, BATCH, eps, isign,
    module.WebNufftModeOrder.Centered, SIGMA, precision,
  );
  const gpuCoefficients = uploadPacked(context, coefficients, precisionName);
  const type2Output = context.createBuffer(type2.outputBytes);
  await type2.execute(gpuPoints, gpuCoefficients, type2Output, BATCH);
  const type2Actual = decodeComplex(await context.download(type2Output), precisionName);
  const type2Expected = referenceType2(shape, points.canonical, coefficients.canonical, isign, BATCH);
  recordAccuracy(results, "type-2", shape.length, precisionName, isign, type2Actual, type2Expected);
}

async function runType3(context, module, dimensions, isign, precisionName, results) {
  const precision = module.WebFftPrecision[precisionName];
  const eps = precisionName === "F32" ? F32_EPS : DF64_EPS;
  const bounds = type3Bounds(dimensions);
  const source = packScalars(intervalPoints(bounds.source, TYPE3_SOURCE_COUNT), precisionName);
  const target = packScalars(intervalPoints(bounds.target, TYPE3_TARGET_COUNT), precisionName);
  const strengths = packComplex(complexValues(BATCH, TYPE3_SOURCE_COUNT, 0.53), precisionName);
  const plan = await context.createNufftType3Plan(
    new Float64Array(bounds.source), new Float64Array(bounds.target),
    TYPE3_SOURCE_COUNT, TYPE3_TARGET_COUNT, BATCH, eps, isign, SIGMA, precision,
  );
  const gpuSource = uploadPacked(context, source, precisionName);
  const gpuTarget = uploadPacked(context, target, precisionName);
  const gpuStrengths = uploadPacked(context, strengths, precisionName);
  const output = context.createBuffer(plan.outputBytes);
  await plan.execute(gpuSource, gpuStrengths, gpuTarget, output, BATCH);
  const actual = decodeComplex(await context.download(output), precisionName);
  const expected = referenceType3(
    source.canonical, target.canonical, strengths.canonical, dimensions, isign, BATCH,
  );
  recordAccuracy(results, "type-3", dimensions, precisionName, isign, actual, expected);
}

function recordAccuracy(results, kind, dimensions, precision, isign, actual, expected) {
  const error = relativeL2(actual, expected);
  const limit = tolerance(kind, dimensions, precision);
  if (!Number.isFinite(error) || error > limit) {
    throw new Error(`${kind} ${dimensions}D ${precision} isign=${isign}: relative L2 ${error} > ${limit}`);
  }
  results.push({ kind, dimensions, precision, isign, relativeL2: error, tolerance: limit });
}

async function runPhaseBoundary(context, module, results) {
  const sourceBounds = new Float64Array([1, 1]);
  const acceptedBounds = new Float64Array([1023.75, 1024.25]);
  const source = packScalars([1, 1], "Df64");
  const target = packScalars([1023.75, 1024, 1024.25], "Df64");
  const strengths = packComplex([0.75, -0.25, -0.125, 0.625], "Df64");
  const plan = await context.createNufftType3Plan(
    sourceBounds, acceptedBounds, 2, 3, 1, DF64_EPS, -1, SIGMA, module.WebFftPrecision.Df64,
  );
  const gpuSource = uploadPacked(context, source, "Df64");
  const gpuTarget = uploadPacked(context, target, "Df64");
  const gpuStrengths = uploadPacked(context, strengths, "Df64");
  const output = context.createBuffer(plan.outputBytes);
  await plan.execute(gpuSource, gpuStrengths, gpuTarget, output, 1);
  const actual = decodeComplex(await context.download(output), "Df64");
  const expected = referenceType3(source.canonical, target.canonical, strengths.canonical, 1, -1, 1);
  const error = relativeL2(actual, expected);
  const limit = 24 * DF64_EPS;
  if (!Number.isFinite(error) || error > limit) {
    throw new Error(`df64 type-3 phase-boundary relative L2 ${error} > ${limit}`);
  }

  let rejection = "";
  try {
    await context.createNufftType3Plan(
      sourceBounds, new Float64Array([1024.75, 1025.25]), 1, 1, 1,
      DF64_EPS, 1, SIGMA, module.WebFftPrecision.Df64,
    );
    throw new Error("df64 type-3 phase bound 1025 was unexpectedly accepted");
  } catch (errorValue) {
    rejection = errorValue?.message ?? String(errorValue);
    if (!rejection.includes("source pre-phase") || !rejection.includes("1025")) throw errorValue;
  }
  results.phaseBoundary = {
    acceptedBound: 1024,
    relativeL2: error,
    tolerance: limit,
    rejectedBound: 1025,
    rejection,
  };
}

async function assertBrowserF64Rejected(context, module) {
  try {
    await context.createNufftType2Plan(
      new Uint32Array([8]), 2, 1, F32_EPS, 1,
      module.WebNufftModeOrder.Centered, SIGMA, module.WebFftPrecision.F64,
    );
  } catch (errorValue) {
    const message = errorValue?.message ?? String(errorValue);
    if (message.includes("SHADER_F64") || message.includes("precision")) return message;
    throw errorValue;
  }
  throw new Error("browser native-f64 NUFFT was unexpectedly accepted");
}

async function runMatrix() {
  const module = await import("../wgpu-web/pkg/wgpu_web.js");
  if (typeof module.default === "function") await module.default();
  const context = await module.WgpuFft.initWithDefaultLimits();
  const limits = {
    maxStorageBufferBindingSize: context.maxStorageBufferBindingSize,
    maxBufferSize: context.maxBufferSize,
    maxComputeWorkgroupStorageSize: context.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: context.maxComputeInvocationsPerWorkgroup,
  };
  const expectedLimits = {
    maxStorageBufferBindingSize: 134217728,
    maxBufferSize: 268435456,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
  };
  for (const [name, expected] of Object.entries(expectedLimits)) {
    if (limits[name] !== expected) throw new Error(`${name}=${limits[name]}, expected ${expected}`);
  }
  if (!context.df64Available || context.df64CanaryWords !== 96) {
    throw new Error(`Tint df64 canary failed: words=${context.df64CanaryWords} ${context.df64CanaryError ?? ""}`);
  }

  const accuracy = [];
  for (const precision of ["F32", "Df64"]) {
    await runType12(context, module, [17], 1, precision, accuracy);
    await runType12(context, module, [8, 12], -1, precision, accuracy);
    await runType12(context, module, [4, 6, 8], 1, precision, accuracy);
    await runType3(context, module, 1, -1, precision, accuracy);
    await runType3(context, module, 2, 1, precision, accuracy);
    await runType3(context, module, 3, -1, precision, accuracy);
  }
  const extra = {};
  await runPhaseBoundary(context, module, extra);
  const f64Rejection = await assertBrowserF64Rejected(context, module);
  return {
    ok: true,
    browser: navigator.userAgent,
    backend: context.backend,
    adapter: {
      name: context.adapterName,
      vendor: context.adapterVendor,
      device: context.adapterDevice,
      deviceType: context.adapterDeviceType,
    },
    limits,
    df64Canary: { cases: 4, exactWords: context.df64CanaryWords, available: context.df64Available },
    matrix: {
      batch: BATCH,
      cases: accuracy,
      passCount: accuracy.length,
      worstF32: Math.max(...accuracy.filter((row) => row.precision === "F32").map((row) => row.relativeL2)),
      worstDf64: Math.max(...accuracy.filter((row) => row.precision === "Df64").map((row) => row.relativeL2)),
    },
    phaseBoundary: extra.phaseBoundary,
    f64Rejection,
    rustRevision: qs.get("rust_revision") ?? "",
    rustWorkingTreeDirty: qs.get("rust_dirty") === "1",
  };
}

async function publish(result) {
  const encoded = encodeURIComponent(JSON.stringify(result));
  document.documentElement.setAttribute("data-wgpu-nufft-phase-d", encoded);
  statusEl.textContent = JSON.stringify(result, null, 2);
  try {
    await fetch("/__phase_d_result", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(result),
    });
  } catch {
    // The DOM marker remains available to the headless fallback.
  }
}

async function run() {
  runButton.disabled = true;
  statusEl.textContent = "Running browser-default NUFFT matrix...";
  try {
    await publish(await runMatrix());
  } catch (error) {
    await publish({
      ok: false,
      error: error?.stack ?? error?.message ?? String(error),
      browser: navigator.userAgent,
      rustRevision: qs.get("rust_revision") ?? "",
      rustWorkingTreeDirty: qs.get("rust_dirty") === "1",
    });
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", run);
if (qs.get("autorun") === "1") run();
