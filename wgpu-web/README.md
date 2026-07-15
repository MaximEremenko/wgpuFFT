# wgpu-web

Minimal `wasm-bindgen` surface for using `wgpu-fft` from browser JavaScript.
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

The upload surface accepts `Float32Array` storage and download returns raw
`Uint8Array` storage:

- F32 complex values are interleaved `re, im` words (8 bytes per element).
- Df64 complex values are `re_hi, re_lo, im_hi, im_lo` words (16 bytes per
  element).
- awaited `execute` covers encode, submit, and queue completion while keeping
  plan creation, upload, and reusable output allocation outside the timed span.
- `exportSnapshot` / awaited `importSnapshot` persist validated shader-source
  and pipeline-key prewarm data; the demo stores it in `localStorage`.
