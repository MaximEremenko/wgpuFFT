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

set "WASM_BINDGEN_TEST_WEBDRIVER_JSON=%~dp0wasm-webdriver.json"
cargo test --target wasm32-unknown-unknown --test wasm_smoke -- --nocapture
exit /b %ERRORLEVEL%
