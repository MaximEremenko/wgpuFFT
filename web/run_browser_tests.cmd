@echo off
setlocal

if not defined CHROMEDRIVER (
    for /f "delims=" %%I in ('where chromedriver.exe 2^>nul') do (
        set "CHROMEDRIVER=%%I"
        goto :driver_found
    )
)

:driver_found
if not defined CHROMEDRIVER (
    echo CHROMEDRIVER is not set and chromedriver.exe was not found on PATH. 1>&2
    echo Set CHROMEDRIVER to a ChromeDriver matching the installed Chrome build. 1>&2
    exit /b 2
)

set "WASM_BINDGEN_TEST_WEBDRIVER_JSON=%~dp0..\webdriver.json"
set "WASM_BINDGEN_TEST_TIMEOUT=300"
set "WGPU_FFT_RUN_BROWSER_LARGE_TESTS=1"
cargo test -p wgpu-fft --target wasm32-unknown-unknown --test wasm_smoke --test wasm_browser_matrix --test wasm_large_routes -- --nocapture
if errorlevel 1 exit /b %ERRORLEVEL%

cargo test -p wgpu-nufft --target wasm32-unknown-unknown --test wasm_browser -- --nocapture
if errorlevel 1 exit /b %ERRORLEVEL%

cargo test -p wgpu-web --target wasm32-unknown-unknown --test wasm_nufft -- --nocapture
exit /b %ERRORLEVEL%
