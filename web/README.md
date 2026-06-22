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

The checked-in Cargo environment setting loads `wasm-webdriver.json`, which
pins the WebGPU-related Chrome flags. The wrapper forces ChromeDriver rather
than allowing `wasm-bindgen-test-runner` to select another WebDriver found on
`PATH` first.
