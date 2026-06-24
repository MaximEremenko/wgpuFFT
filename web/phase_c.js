// Browser Phase C harness contract for the wasm-bindgen wrapper:
//   default init(); WgpuFft.init(); await context.createPlan(...);
//   context.upload(Float32Array); context.createBuffer(byteLength);
//   await plan.execute(input, output); await context.download(buffer);
//   context.exportSnapshot(); await context.importSnapshot(json).

const SNAPSHOT_KEY = "wgpu-fft.pipeline-cache.v1";
const JS_SOURCE_REVISION = "fa45c93f524a69a96c9f55acfad865226bfccd29";
const CASE = Object.freeze({ n: 4096, batch: 1024, warmups: 5, runs: 3, iterations: 20 });
const qs = new URLSearchParams(location.search);

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const initButton = document.getElementById("init");
const snapshotButton = document.getElementById("snapshot");
const benchButton = document.getElementById("bench");

function log(line) {
  logEl.textContent += `${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text, kind = "") {
  statusEl.className = kind;
  statusEl.textContent = text;
}

function makeImpulseInput() {
  const data = new Float32Array(CASE.n * CASE.batch * 2);
  for (let b = 0; b < CASE.batch; b += 1) data[2 * b * CASE.n] = 1;
  return data;
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function stderr(xs) {
  if (xs.length < 2) return 0;
  const avg = mean(xs);
  const variance = xs.reduce((s, x) => s + (x - avg) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(variance / xs.length);
}

function summarize(name, samples) {
  return { name, samplesMs: samples, avgMs: mean(samples), stderrMs: stderr(samples) };
}

function adapterInfoRecord(info) {
  if (!info) return null;
  return {
    vendor: String(info.vendor ?? ""),
    architecture: String(info.architecture ?? ""),
    device: String(info.device ?? ""),
    description: String(info.description ?? ""),
  };
}

function parseBrowserVendorId(vendor) {
  const normalized = String(vendor ?? "").trim().toLowerCase();
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function parseBrowserDeviceId(device) {
  const normalized = String(device ?? "").trim().toLowerCase();
  if (!normalized) return null;
  const parsed = Number(normalized);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function adapterIdentityEvidence(rust, jsInfo) {
  const jsVendor = parseBrowserVendorId(jsInfo?.vendor);
  const rustVendor = rust.adapter.vendor || null;
  if (jsVendor !== null && rustVendor !== null && jsVendor !== rustVendor) {
    throw new Error(`Rust/JS adapter vendor mismatch: ${rustVendor} vs ${jsVendor}`);
  }
  const jsDevice = parseBrowserDeviceId(jsInfo?.device);
  const rustDevice = rust.adapter.device || null;
  if (jsDevice !== null && rustDevice !== null && jsDevice !== rustDevice) {
    throw new Error(`Rust/JS adapter device mismatch: ${rustDevice} vs ${jsDevice}`);
  }
  const vendorObservable = jsVendor !== null && rustVendor !== null;
  const deviceObservable = jsDevice !== null && rustDevice !== null;
  const observable = vendorObservable && deviceObservable;
  return {
    observable,
    match: observable ? true : null,
    note: observable
      ? "browser and wgpu vendor/device identifiers match"
      : vendorObservable
        ? "vendor identifiers match, but Chrome privacy omitted a device identifier"
        : "Chrome privacy omitted at least one adapter vendor/device identifier",
  };
}

function readStoredSnapshotFromFreshDocument(key) {
  return new Promise((resolve, reject) => {
    const iframe = document.createElement("iframe");
    iframe.hidden = true;
    const timeout = setTimeout(() => finish(new Error("fresh-document localStorage probe timed out")), 10_000);
    function finish(error, value) {
      clearTimeout(timeout);
      removeEventListener("message", onMessage);
      iframe.remove();
      if (error) reject(error); else resolve(value);
    }
    function onMessage(event) {
      if (
        event.origin !== location.origin
        || event.source !== iframe.contentWindow
        || event.data?.type !== "wgpu-fft-storage-probe"
        || event.data?.key !== key
      ) return;
      finish(null, event.data.value);
    }
    addEventListener("message", onMessage);
    iframe.src = `./storage_probe.html?key=${encodeURIComponent(key)}&nonce=${Date.now()}`;
    document.body.appendChild(iframe);
  });
}

function assertImpulse(output, label) {
  if (!(output instanceof Float32Array)) {
    if (output instanceof ArrayBuffer) {
      output = new Float32Array(output);
    } else if (ArrayBuffer.isView(output)) {
      output = new Float32Array(output.buffer, output.byteOffset, output.byteLength / Float32Array.BYTES_PER_ELEMENT);
    } else {
      throw new Error(`${label}: download returned an unsupported value`);
    }
  }
  const expectedLength = CASE.n * CASE.batch * 2;
  if (output.length !== expectedLength) {
    throw new Error(`${label}: output length ${output.length}, expected ${expectedLength}`);
  }
  let worst = 0;
  for (let i = 0; i < output.length; i += 2) {
    worst = Math.max(worst, Math.abs(output[i] - 1), Math.abs(output[i + 1]));
  }
  if (worst > 2e-5) throw new Error(`${label}: impulse FFT max absolute error ${worst}`);
  return worst;
}

async function loadRustModule() {
  const module = await import("../wgpu-web/pkg/wgpu_web.js");
  if (typeof module.default === "function") await module.default();
  if (!module.WgpuFft || typeof module.WgpuFft.init !== "function") {
    throw new Error("wgpu-web wrapper must export WgpuFft.init()");
  }
  return module;
}

async function newRustContext(module) {
  const context = await module.WgpuFft.init();
  for (const name of [
    "createPlan", "upload", "createBuffer", "download", "exportSnapshot", "importSnapshot",
  ]) {
    if (typeof context[name] !== "function") throw new Error(`WgpuFft context is missing ${name}()`);
  }
  return context;
}

function buildRequiredLimitsFromAdapter(adapter) {
  const limits = adapter?.limits;
  if (!limits) return null;
  const names = [
    "maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
  ];
  const required = {};
  for (const name of names) {
    const value = limits[name];
    if (Number.isFinite(value) && value > 0) required[name] = Math.floor(value);
  }
  if (required.maxBufferSize && required.maxStorageBufferBindingSize) {
    required.maxStorageBufferBindingSize = Math.min(
      required.maxStorageBufferBindingSize,
      required.maxBufferSize,
    );
  }
  return Object.keys(required).length ? required : null;
}

async function requestMatchingJsDevice() {
  if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Chrome did not return a WebGPU adapter");
  // Mirror WgpuFft.init(): request the adapter's supported limits without
  // optional features, then fall back to browser defaults if Chrome rejects it.
  const requiredLimits = buildRequiredLimitsFromAdapter(adapter);
  let device;
  let limitsRequestMode = "adapter-max";
  try {
    device = await adapter.requestDevice(requiredLimits ? { requiredLimits } : {});
  } catch {
    limitsRequestMode = "default-fallback";
    device = await adapter.requestDevice();
  }
  const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
  return { adapter, device, info, limitsRequestMode, requestedLimits: requiredLimits };
}

async function createRustBench(module, inputData) {
  const context = await newRustContext(module);
  const plan = await context.createPlan(
    CASE.n,
    CASE.batch,
    module.WebFftDirection.Forward,
    module.WebFftPrecision.F32,
    module.WebFftNormalization.None,
  );
  const input = context.upload(inputData);
  const output = context.createBuffer(plan.outputBytes);
  return {
    name: "wgpu-fft Rust/Wasm",
    adapter: {
      name: context.adapterName,
      vendor: context.adapterVendor,
      device: context.adapterDevice,
      deviceType: context.adapterDeviceType,
      driver: context.adapterDriver,
      driverInfo: context.adapterDriverInfo,
    },
    backend: context.backend,
    route: plan.route,
    limits: {
      maxBufferSize: context.maxBufferSize,
      maxStorageBufferBindingSize: context.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: context.maxComputeWorkgroupStorageSize,
    },
    async assertAliasRejected() {
      try {
        await plan.execute(input, input);
      } catch (error) {
        if (String(error).includes("must be distinct buffers")) return true;
        throw error;
      }
      throw new Error("out-of-place wrapper accepted the same input/output buffer");
    },
    async execute() { await plan.execute(input, output); },
    async download() { return context.download(output); },
    destroy() {
      plan?.free?.(); input?.free?.(); output?.free?.(); context?.free?.();
    },
  };
}

async function createJsBench(device, inputData) {
  const lib = await import("../../WebGPU-FFT/src/index.js");
  const plan = lib.createPlan(device, {
    type: "c2c",
    shape: [CASE.n],
    batch: CASE.batch,
    direction: "forward",
    inPlace: false,
    normalize: "none",
    layout: { interleavedComplex: true },
    precision: "f32",
  });
  const input = lib.uploadComplex(device, inputData);
  const output = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  return {
    name: "WebGPU-FFT JavaScript",
    async execute() {
      const encoder = device.createCommandEncoder();
      plan.exec(encoder, { input, output });
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    },
    async download() {
      const readback = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(output, 0, readback, 0, inputData.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      readback.destroy();
      return result;
    },
    destroy() { plan.destroy(); input.destroy(); output.destroy(); },
  };
}

async function measureBlock(implementation) {
  const start = performance.now();
  for (let i = 0; i < CASE.iterations; i += 1) await implementation.execute();
  return (performance.now() - start) / CASE.iterations;
}

async function runBenchmark(state) {
  log(`Benchmark case: C2C forward N=${CASE.n}, batch=${CASE.batch}, f32`);
  log(`Timing: ${CASE.runs} alternating runs x ${CASE.iterations} iterations; ${CASE.warmups} warmups`);
  log("Timed span: encode + one submit + queue completion; plan/upload/allocation/download excluded");
  const input = makeImpulseInput();
  const rust = await createRustBench(state.rustModule, input);
  const js = await createJsBench(state.jsDevice, input);
  try {
    const jsLimits = {
      maxStorageBufferBindingSize: state.jsDevice.limits.maxStorageBufferBindingSize,
      maxBufferSize: state.jsDevice.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: state.jsDevice.limits.maxComputeWorkgroupStorageSize,
    };
    if (
      rust.limits.maxBufferSize !== jsLimits.maxBufferSize
      || rust.limits.maxStorageBufferBindingSize !== jsLimits.maxStorageBufferBindingSize
      || rust.limits.maxComputeWorkgroupStorageSize !== jsLimits.maxComputeWorkgroupStorageSize
    ) {
      throw new Error(`Rust/JS active limit mismatch: ${JSON.stringify(rust.limits)} vs ${JSON.stringify(jsLimits)}`);
    }
    const adapterIdentity = adapterIdentityEvidence(rust, state.adapterInfo);
    const aliasRejected = await rust.assertAliasRejected();
    log(`Rust adapter/backend: ${JSON.stringify(rust.adapter)} / ${rust.backend}; route=${rust.route}`);
    log(`Adapter identity: ${adapterIdentity.note}`);
    for (let i = 0; i < CASE.warmups; i += 1) {
      await rust.execute();
      await js.execute();
    }
    const samples = { rust: [], js: [] };
    for (let run = 0; run < CASE.runs; run += 1) {
      const order = run % 2 === 0 ? [["rust", rust], ["js", js]] : [["js", js], ["rust", rust]];
      for (const [key, implementation] of order) {
        const value = await measureBlock(implementation);
        samples[key].push(value);
        log(`run ${run + 1} ${implementation.name}: ${value.toFixed(4)} ms/transform`);
      }
    }
    const rustWorst = assertImpulse(await rust.download(), rust.name);
    const jsWorst = assertImpulse(await js.download(), js.name);
    const rustSummary = summarize(rust.name, samples.rust);
    const jsSummary = summarize(js.name, samples.js);
    const ratio = rustSummary.avgMs / jsSummary.avgMs;
    log(`${rust.name}: ${rustSummary.avgMs.toFixed(4)} +/- ${rustSummary.stderrMs.toFixed(4)} ms`);
    log(`${js.name}: ${jsSummary.avgMs.toFixed(4)} +/- ${jsSummary.stderrMs.toFixed(4)} ms`);
    log(`Rust/JS ratio: ${ratio.toFixed(3)}x; correctness worst abs ${rustWorst}/${jsWorst}`);
    return { case: CASE, rust: rustSummary, js: jsSummary, rustOverJs: ratio,
      rustAdapter: rust.adapter, jsAdapter: state.adapterInfo,
      adapterIdentity, rustBackend: rust.backend, rustRoute: rust.route,
      limitsRequestMode: state.limitsRequestMode, activeLimits: jsLimits,
      aliasRejected, rustWorstAbs: rustWorst, jsWorstAbs: jsWorst };
  } finally {
    rust.destroy();
    js.destroy();
  }
}

async function runSnapshotRoundTrip(state) {
  const cold = await newRustContext(state.rustModule);
  const adapter = cold.adapterName;
  const backend = cold.backend;
  log(`Rust snapshot context: ${adapter} / ${backend}`);
  const coldStart = performance.now();
  const coldPlan = await cold.createPlan(
    CASE.n,
    CASE.batch,
    state.rustModule.WebFftDirection.Forward,
    state.rustModule.WebFftPrecision.F32,
    state.rustModule.WebFftNormalization.None,
  );
  const coldPlanMs = performance.now() - coldStart;
  const json = cold.exportSnapshot();
  localStorage.setItem(SNAPSHOT_KEY, json);
  const saved = await readStoredSnapshotFromFreshDocument(SNAPSHOT_KEY);
  if (saved !== json) {
    throw new Error("fresh-document pipeline snapshot localStorage round trip changed bytes");
  }

  const warm = await newRustContext(state.rustModule);
  const importStart = performance.now();
  const normalized = await warm.importSnapshot(saved);
  const importMs = performance.now() - importStart;
  if (normalized !== saved) throw new Error("pipeline snapshot import did not preserve normalized JSON bytes");
  const warmStart = performance.now();
  const warmPlan = await warm.createPlan(
    CASE.n,
    CASE.batch,
    state.rustModule.WebFftDirection.Forward,
    state.rustModule.WebFftPrecision.F32,
    state.rustModule.WebFftNormalization.None,
  );
  const warmPlanMs = performance.now() - warmStart;

  const inputData = makeImpulseInput();
  const input = warm.upload(inputData);
  const output = warm.createBuffer(warmPlan.outputBytes);
  await warmPlan.execute(input, output);
  const worstAbs = assertImpulse(await warm.download(output), "snapshot-restored Rust/Wasm");
  log(`Snapshot key=${SNAPSHOT_KEY}, JSON bytes=${new TextEncoder().encode(json).byteLength}`);
  log(`Cold plan=${coldPlanMs.toFixed(3)} ms; import=${importMs.toFixed(3)} ms; restored plan=${warmPlanMs.toFixed(3)} ms`);
  log(`Restored-plan impulse max abs=${worstAbs}`);

  coldPlan?.free?.(); cold?.free?.(); warmPlan?.free?.(); input?.free?.(); output?.free?.(); warm?.free?.();
  return { storage: "localStorage", persistenceProbe: "fresh-same-origin-document", key: SNAPSHOT_KEY,
    jsonBytes: new TextEncoder().encode(json).byteLength,
    adapter, backend, coldPlanMs, importMs, restoredPlanMs: warmPlanMs, worstAbs };
}

async function initialize() {
  setStatus("initializing Chrome WebGPU and Rust/Wasm...");
  const [rustModule, js] = await Promise.all([loadRustModule(), requestMatchingJsDevice()]);
  snapshotButton.disabled = false;
  benchButton.disabled = false;
  const info = adapterInfoRecord(js.info);
  log(`Chrome: ${navigator.userAgent}`);
  log(`JS adapter: ${JSON.stringify(info)}`);
  log(`JS reference revision: ${JS_SOURCE_REVISION}`);
  log(`Rust revision supplied by runner: ${qs.get("rust_revision") || "unknown"}${qs.get("rust_dirty") === "1" ? " + uncommitted Phase C changes" : ""}`);
  log("Rust adapter/backend and selected route are reported after its context and plan are created.");
  log(`JS limit request: ${js.limitsRequestMode}`);
  log(`Active matching limits: bind=${js.device.limits.maxStorageBufferBindingSize}, buffer=${js.device.limits.maxBufferSize}, workgroupStorage=${js.device.limits.maxComputeWorkgroupStorageSize}`);
  setStatus("ready", "pass");
  return { rustModule, jsDevice: js.device, adapterInfo: info, limitsRequestMode: js.limitsRequestMode };
}

let statePromise;
function ensureInitialized() {
  if (!statePromise) statePromise = initialize();
  return statePromise;
}

async function publish(result) {
  const payload = {
    ok: true,
    chrome: navigator.userAgent,
    rustRevision: qs.get("rust_revision") || "unknown",
    rustWorkingTreeDirty: qs.get("rust_dirty") === "1",
    jsRevision: JS_SOURCE_REVISION,
    ...result,
  };
  const json = JSON.stringify(payload);
  document.documentElement.dataset.wgpuFftPhaseC = encodeURIComponent(json);
  try {
    await fetch("/__phase_c_result", { method: "POST", headers: { "content-type": "application/json" }, body: json });
  } catch { /* DOM marker remains available to the headless runner. */ }
}

async function runAction(kind) {
  initButton.disabled = snapshotButton.disabled = benchButton.disabled = true;
  setStatus(`running ${kind}...`);
  try {
    const state = await ensureInitialized();
    const result = {};
    if (kind === "snapshot" || kind === "all") result.snapshot = await runSnapshotRoundTrip(state);
    if (kind === "bench" || kind === "all") result.benchmark = await runBenchmark(state);
    setStatus(`${kind} passed`, "pass");
    await publish(result);
  } catch (error) {
    const message = String(error?.stack ?? error);
    log(message);
    setStatus(`${kind} failed: ${String(error?.message ?? error)}`, "fail");
    await publish({ ok: false, error: message });
    throw error;
  } finally {
    initButton.disabled = false;
    if (statePromise) snapshotButton.disabled = benchButton.disabled = false;
  }
}

initButton.addEventListener("click", () => ensureInitialized().catch((e) => log(String(e?.stack ?? e))));
snapshotButton.addEventListener("click", () => runAction("snapshot").catch(() => {}));
benchButton.addEventListener("click", () => runAction("bench").catch(() => {}));

const autorun = qs.get("autorun");
if (["snapshot", "bench", "all"].includes(autorun)) runAction(autorun).catch(() => {});
