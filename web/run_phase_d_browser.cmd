@echo off
setlocal
python "%~dp0run_phase_d_browser.py" %*
exit /b %ERRORLEVEL%
