# FFT browser tests

These standalone browser tests exercise the `wgpu-fft` crate through
`wasm-bindgen-test-runner`, ChromeDriver, and Chrome's WebGPU implementation.
Install the matching `wasm-bindgen-cli` version and the
`wasm32-unknown-unknown` target, then point `CHROMEDRIVER` at a ChromeDriver
matching the installed Chrome build:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.120 --locked
$env:CHROMEDRIVER = 'C:\path\to\chromedriver.exe'
web\run_browser_tests.cmd
```

The runner loads the checked-in root `webdriver.json`, which pins the
WebGPU-related Chrome flags. It explicitly selects ChromeDriver so
`wasm-bindgen-test-runner` does not choose another WebDriver found earlier on
`PATH`.

The wrapper runs these FFT-only test targets:

- `wasm_smoke`: basic browser execution.
- `wasm_browser_matrix`: the browser-default FFT correctness matrix and exact
  df64 canaries.
- `wasm_large_routes`: natural four-step and segmented-volume cases.

The large-route cases allocate roughly 1 GiB of transient GPU resources. The
wrapper opts into them with `WGPU_FFT_RUN_BROWSER_LARGE_TESTS=1`; ordinary
`cargo test` runs do not enable them.
