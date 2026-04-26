@echo off
setlocal

title LMDIS Launcher
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo ============================================================
echo LMDIS Launcher
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" goto :venv_ready

echo [1/3] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 goto :err_nopython

for /f "delims=" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo        Found %PY_VER%
echo.

echo [2/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto :err_venv
echo        Virtual environment created.
echo.

echo [3/3] Installing dependencies from backend\requirements.txt...
".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 goto :err_deps
echo        Dependencies installed.
echo.
goto :launch

:venv_ready
for /f "delims=" %%v in ('".venv\Scripts\python.exe" --version 2^>^&1') do set "PY_VER=%%v"
echo [1/1] Using existing virtual environment.
echo        %PY_VER%
echo.

:launch
echo Starting application...
echo Browser URL: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" launcher.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
	echo LMDIS session ended.
) else (
	echo LMDIS exited with code %EXIT_CODE%.
)
echo Press any key to close this window.
pause >nul
exit /b %EXIT_CODE%

:err_nopython
echo ERROR: Python was not found on PATH.
echo Install Python 3.10 or newer, then run start.bat again.
goto :fatal

:err_venv
echo ERROR: Failed to create the virtual environment.
echo Verify that Python is installed correctly and try again.
goto :fatal

:err_deps
echo ERROR: Failed to install backend dependencies.
echo Check backend\requirements.txt and your network connection.
goto :fatal

:fatal
echo.
echo Startup failed. Press any key to exit.
pause >nul
exit /b 1
