# wgpu-fft-web

JavaScript bindings of `wgpu-fft` for the browser, built with `wasm-bindgen`:
N-dimensional complex-to-complex plans on WebGPU, whose inputs and outputs
stay on the GPU until an explicit `download`, and `cpuFft`, the host-memory
`CpuFftPlan` in `f64`, for pages or workers without WebGPU. The structure
follows `wgpu-web` in wgpuNUFFT.

## Loading

`dist/wgpu_fft_web.js` is one classic script that defines `wgpuFftWeb`. It
embeds the WebAssembly module, so it works in any page or worker, including
pages opened from disk:

```html
<script src="wgpu_fft_web.js"></script>
<script>
  wgpuFftWeb.load().then(async ({ WgpuFft, cpuFft }) => {
    const gpu = await WgpuFft.init();
    // ...
  });
</script>
```

The file is not checked in. Build it with
`python wgpu-fft-web/build_standalone.py`, which needs the
`wasm32-unknown-unknown` target and `wasm-bindgen-cli` in the version of the
`wasm-bindgen` crate in `Cargo.lock`, currently 0.2.129
(`cargo install wasm-bindgen-cli --version 0.2.129 --locked`;
`--wasm-bindgen PATH` picks another binary). CI builds it on every push as
the `wgpu_fft_web` artifact of the run and attaches it to every release.

## Example

A forward 3-D transform of 64 x 64 x 32 complex values on the GPU:

```js
const { WgpuFft, WebFftDirection, WebFftPrecision, WebFftNormalization } = await wgpuFftWeb.load();
const gpu = await WgpuFft.init();
const plan = await gpu.createPlan(
  new Uint32Array([64, 64, 32]), // shape, axis 0 fastest
  1,                             // batch
  WebFftDirection.Forward,
  WebFftPrecision.F32,
  WebFftNormalization.None,
);
const input = gpu.upload(values);          // Float32Array of interleaved re, im
const output = gpu.createBuffer(plan.outputBytes);
await plan.execute(input, output);
const bytes = await gpu.download(output);
const spectrum = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
for (const object of [input, output, plan, gpu]) object.free();
```

The same transform in `f64` without WebGPU:

```js
const { cpuFft, WebFftDirection, WebFftNormalization } = await wgpuFftWeb.load();
const spectrum = cpuFft(new Uint32Array([64, 64, 32]), valuesF64,
  WebFftDirection.Forward, WebFftNormalization.None); // Float64Array, interleaved
```

## API

| Call | Description |
|---|---|
| `WgpuFft.init()` | Opens WebGPU with the adapter's maximum limits, falling back to the defaults if the browser rejects them. |
| `WgpuFft.initWithDefaultLimits()` | Opens WebGPU with exactly the default limits. |
| `WgpuFft.initFallback()` | Opens the browser's software adapter, such as SwiftShader. |
| `createPlan(shape, batch, direction, precision, normalization)` | A reusable C2C plan over every axis of `shape` (a `Uint32Array`, axis 0 fastest); `execute(input, output)`. |
| `upload(Float32Array)`, `uploadDf64(Float64Array)`, `createBuffer(byteLength)` | Create GPU buffers. |
| `download(buffer)` | Resolves to the buffer's bytes as a `Uint8Array`. |
| `cpuFft(shape, data, direction, normalization)` | C2C transform of a `Float64Array` of interleaved `re, im` pairs in host memory (`f64`, no WebGPU); returns a `Float64Array`. |
| `adapterName`, `backend`, `maxBufferSize`, `maxStorageBufferBindingSize`, `df64Available` | Describe the device. |

Plans report `inputBytes`, `outputBytes` and `workspaceBytes`. The forward
kernel is `exp(-2 pi i jk / n)`; `Inverse` changes its sign, and the
normalization places the `1 / n` factor.

## Data layout and precision

`F32` complex values are interleaved `re, im` words; `Df64` values are
`re_hi, re_lo, im_hi, im_lo` (`uploadDf64` splits each `f64`). WebGPU has no
64-bit float shaders: `F64` plans reach `wgpu-fft`, which reports that, and
`cpuFft` is the `f64` route. `init()` runs `wgpu-fft`'s df64 canary suite;
`Df64` plans are refused when it fails, while `F32` keeps working.

## Releasing GPU memory

JavaScript garbage collection does not see GPU memory, so call `free()` on
plans, buffers and the context when done. Freeing a buffer destroys its GPU
allocation at once; freeing the context and everything created from it
destroys the device.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
