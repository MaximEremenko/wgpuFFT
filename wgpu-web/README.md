# wgpu-web

Minimal `wasm-bindgen` surface for using `wgpu-fft` and `wgpu-nufft` from
browser JavaScript.
The API keeps uploaded inputs, outputs, and reusable plans GPU-resident until an
explicit `download` call.

Build the package from the repository root:

```powershell
wasm-pack build wgpu-web --target web --out-dir pkg
python -m http.server 8000 --directory wgpu-web
```

Then open `http://localhost:8000/demo/` in a WebGPU-capable browser. Generated
`pkg/` contents are intentionally ignored.

`WgpuFft.init()` runs `wgpu-fft`'s complete 96-word double-float invariant suite.
If Chrome's Tint compiler or its GPU backend changes the required arithmetic,
F32 remains usable while Df64 plan creation is rejected with the canary failure.
Native F64 is passed through to `wgpu-fft`; browsers return its structured
`device-missing-shader-f64` error.

`WgpuFft.initWithDefaultLimits()` requests no features and exactly WebGPU's
default limits while running the same canary. It exists for correctness testing
and sites that want the most conservative portable device contract; ordinary
`init()` retains the adapter-max-first behavior.

The upload surface accepts `Float32Array` storage and download returns raw
`Uint8Array` storage:

- F32 complex values are interleaved `re, im` words (8 bytes per element).
- Df64 complex values are `re_hi, re_lo, im_hi, im_lo` words (16 bytes per
  element).
- awaited `execute` covers encode, submit, and queue completion while keeping
  plan creation, upload, and reusable output allocation outside the timed span.
- `exportSnapshot` / awaited `importSnapshot` persist validated shader-source
  and pipeline-key prewarm data; the demo stores it in `localStorage`.

## NUFFT surface

The context exposes `createNufftType1Plan`, `createNufftType2Plan`, and
`createNufftType3Plan`. Type-1/type-2 mode shapes are passed as `Uint32Array`;
type-3 bounds are flattened `Float64Array` values in
`[lower0, upper0, lower1, upper1, ...]` order. The Rust future copies these
arrays before awaiting GPU work, but JavaScript callers must retain them
unchanged until the returned plan-creation Promise settles.

Each plan fixes its point counts and batch capacity and exposes the required
full-capacity byte sizes. `execute(..., activeBatch)` accepts any nonzero active
batch not exceeding that capacity. Coordinates are point-major, while complex
strengths, coefficients, and outputs are transform-major. Plans and all buffer
handles enforce WebGPU-device identity and reject aliased input/output roles.
GPU buffers are opaque at encoding time, so callers must enforce the coordinate
contract: type-1/type-2 points must be finite and inside `[-3*pi, 3*pi]`, while
type-3 source and target values must be finite and remain inside the intervals
supplied when the plan was created.

F32 coordinates and complex values use `upload(Float32Array)`. For Df64,
`uploadDf64(Float64Array)` rounds each scalar to an `f32` high word and residual
low word. Thus point coordinates become `[x_hi, x_lo, ...]`, and interleaved
complex `[re, im]` values become
`[re_hi, re_lo, im_hi, im_lo]`. Df64 plan creation is available only after the
browser compiler passes all 96 arithmetic canary words.
