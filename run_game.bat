@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title Bluff Dice - One Click Launcher

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"

if not exist "%PROJECT_DIR%web_game\app.py" (
  for /d %%D in ("%SCRIPT_DIR%bluff-test*") do (
    if exist "%%~fD\web_game\app.py" set "PROJECT_DIR=%%~fD"
  )
)

if not exist "%PROJECT_DIR%web_game\app.py" (
  echo [ERROR] Could not find web_game\app.py.
  echo Please put this script in the project folder and run it again.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"

set "VENV_DIR=%LOCALAPPDATA%\BluffDice\venv-py310"

echo ========================================
echo Bluff Dice - One Click Launcher
echo Project: %PROJECT_DIR%
echo ========================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.10 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    set "PY_BOOT=py -3.10"
  ) else (
    echo [ERROR] Python 3.10 was not found by the Python launcher.
    echo This project needs Python 3.10 because its bundled .pyc files were built for 3.10.
    echo Please install Python 3.10, then run this script again.
    pause
    exit /b 1
  )
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python 3.10, then run this script again.
    pause
    exit /b 1
  )
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] The available python command is not Python 3.10.
    echo This project needs Python 3.10 because its bundled .pyc files were built for 3.10.
    pause
    exit /b 1
  )
  set "PY_BOOT=python"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [1/3] Creating virtual environment in a short Windows path...
  echo Venv: %VENV_DIR%
  %PY_BOOT% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

set "PYTHON=%VENV_DIR%\Scripts\python.exe"

echo [2/3] Installing or updating dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to update pip.
  pause
  exit /b 1
)

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install dependencies.
  echo Check your network connection, then run this script again.
  pause
  exit /b 1
)

if "%PORT%"=="" set "PORT=5000"
if "%AI_MODEL_MODE%"=="" set "AI_MODEL_MODE=online"
if "%BLUFF_MODEL_URL%"=="" set "BLUFF_MODEL_URL=https://gitee.com/nuan4652/bluff-test-1/releases/download/v1.0.0/dmc_v5_best.pth"
set "TRAINING_ENABLED=true"
set "FLASK_ENV=production"
set "URL=http://127.0.0.1:%PORT%/"

echo.
echo [3/3] Starting game server...
echo Opening %URL% when the server is ready...
echo Press Ctrl+C in this window to stop the server.
echo.

start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$url='%URL%'; for ($i=0; $i -lt 180; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2 | Out-Null; Start-Process $url; exit } catch { Start-Sleep -Seconds 1 } }; Start-Process $url"
"%PYTHON%" web_game\app.py

echo.
echo Server stopped.
pause
