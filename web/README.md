# Browser test harness

The browser tests use `wasm-bindgen-test-runner`, ChromeDriver, and Chrome's
WebGPU implementation. Install the matching `wasm-bindgen-cli` version and the
`wasm32-unknown-unknown` target, then point `CHROMEDRIVER` at a ChromeDriver
matching the installed Chrome build:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.120 --locked
$env:CHROMEDRIVER = 'C:\path\to\chromedriver.exe'
web\run_browser_tests.cmd
```

The runner automatically loads the checked-in root `webdriver.json`, which pins
the WebGPU-related Chrome flags. The wrapper forces ChromeDriver rather than
allowing `wasm-bindgen-test-runner` to select another WebDriver found on `PATH`
first.

The wrapper runs the FFT smoke test, browser-default FFT correctness matrix,
exact df64 canaries, natural four-step and segmented-volume cases, the complete
browser NUFFT matrix, and a website-wrapper NUFFT smoke test. The large FFT
cases allocate roughly 1 GiB of transient GPU resources and are therefore
enabled by the wrapper's `WGPU_FFT_RUN_BROWSER_LARGE_TESTS=1` setting instead
of ordinary workspace test commands.

## Pipeline-cache demo and Rust/Wasm vs JavaScript comparison

After building the `wgpu-web` package, run the Phase C page through the same
Chrome/WebGPU setup:

```powershell
web\run_phase_c_browser.cmd --mode all
```

For an archive-ready machine record, add for example
`--output benchmark-results/2026-07-15-browser/phase-c-result.json`.

The runner builds `wgpu-web` for `wasm32-unknown-unknown`, runs `wasm-bindgen`,
serves this repository and the sibling `WebGPU-FFT` checkout from one origin,
and pins the JavaScript reference revision to
`fa45c93f524a69a96c9f55acfad865226bfccd29`. It attempts headless Chrome first
with a 30-second callback deadline, then cleans up that process tree and falls
back to the headed app-window pattern proven by the JavaScript library's
harness. `--headed-only` skips the probe on machines where headless WebGPU is
known to be unavailable.

The short comparison is one out-of-place f32 C2C forward transform at N=4096,
batch=1024. Plans, buffer allocation, upload, and download are outside the timed
region. Each timed iteration includes command encoding, one queue submission,
and `onSubmittedWorkDone`; both implementations reuse their buffers. Five
warmups precede three alternating 20-iteration blocks, reported as average
milliseconds per transform plus standard error across block means. This is an
end-to-end library comparison under the same Chrome/Tint compiler and GPU, not a
claim that the generated shaders are identical. Both sides first request the
adapter's supported limits without optional features and fall back to browser
defaults only if Chrome rejects that request; the runner records the active
limits and the Rust route.

The cache demo creates a plan, exports the versioned JSON snapshot to
`localStorage` under `wgpu-fft.pipeline-cache.v1`, reads it from a fresh
same-origin document, imports it into a fresh WebGPU context, recreates the
plan, executes it, and validates an impulse transform.

## Browser NUFFT matrix

The Phase D runner tests the JavaScript-facing type-1, type-2, and type-3
surface in one through three dimensions, with batched F32 and Df64 data, at
exact WebGPU default limits:

```powershell
web\run_phase_d_browser.cmd --headed-only `
  --output benchmark-results/2026-07-15-browser/phase-d-result.json
```

It rebuilds `wgpu-web`, generates the browser module with `wasm-bindgen`, and
launches Chrome directly. The page compares every result with an independent
JavaScript f64 NDFT, verifies all 96 df64 canary words, checks the inclusive
df64 type-3 phase bound at 1024 and structured rejection at 1025, and confirms
that browser native F64 remains unavailable. Omit `--headed-only` to try a
bounded 30-second headless probe before the headed fallback.
