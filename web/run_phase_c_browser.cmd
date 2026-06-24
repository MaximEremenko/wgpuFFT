@echo off
setlocal
python "%~dp0run_phase_c_browser.py" %*
exit /b %ERRORLEVEL%
