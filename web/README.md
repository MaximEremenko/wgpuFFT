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

The wrapper runs the smoke test, browser-default correctness matrix, exact df64
canaries, and the natural four-step and segmented-volume cases. The latter
allocates roughly 1 GiB of transient GPU resources and is therefore enabled by
the wrapper's `WGPU_FFT_RUN_BROWSER_LARGE_TESTS=1` setting instead of ordinary
workspace test commands.
