@echo off
REM ==============================================================================
REM JARVIS AI System - Easy Run Script for Windows
REM Supports: Ollama, Pinokio, and Local Inference
REM ==============================================================================

setlocal enabledelayedexpansion

REM Script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Print banner
echo.
echo ============================================================
echo                                                       
echo     J  A  R  V  I  S    A  I    S  Y  S  T  E  M       
echo                                                       
echo              Easy Run Script v1.0 (Windows)           
echo              Ollama ^| Pinokio ^| Local Inference       
echo ============================================================
echo.

REM Create logs directory
if not exist logs mkdir logs

REM Check for Node.js
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js from: https://nodejs.org
    pause
    exit /b 1
)

REM Install Node dependencies if needed
if not exist node_modules (
    echo [INFO] Installing Node dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check backend availability
set BACKEND_AVAILABLE=0

REM Check Ollama
where ollama >nul 2>nul
if not errorlevel 1 (
    set BACKEND_AVAILABLE=1
    echo [OK] Ollama detected
)

REM Check Python (for local inference)
where python >nul 2>nul
if not errorlevel 1 (
    set BACKEND_AVAILABLE=1
    echo [OK] Python detected
    if not exist requirements.txt (
        echo [WARNING] requirements.txt not found
    )
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set BACKEND_AVAILABLE=1
        echo [OK] Python detected (as py command)
    )
)

if %BACKEND_AVAILABLE%==0 (
    echo [WARNING] No AI backend detected
    echo [INFO] You can still run in demo mode
)

REM Show menu
echo.
echo Available Options:
echo.
echo 1. Start with Ollama (if installed)
echo 2. Start with Local Inference (if Python available)
echo 3. Start in Demo Mode (no AI backend)
echo 4. Exit
echo.
set /p CHOICE="Select option (1-4): "

if "%CHOICE%"=="1" goto OLLAMA
if "%CHOICE%"=="2" goto LOCAL
if "%CHOICE%"=="3" goto DEMO
if "%CHOICE%"=="4" goto END

echo [ERROR] Invalid choice
pause
exit /b 1

:OLLAMA
echo.
echo [INFO] Setting up Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Ollama is not installed
    echo Please install from: https://ollama.ai/download
    pause
    exit /b 1
)

echo [INFO] Checking Ollama service...
ollama list >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Ollama service may not be running
    echo Try running: ollama serve
    echo.
)

set /p MODEL_NAME="Enter model name (default: llama3.2): "
if "%MODEL_NAME%"=="" set MODEL_NAME=llama3.2

echo [INFO] Using model: %MODEL_NAME%
set BACKEND=ollama
goto START_SERVER

:LOCAL
echo.
echo [INFO] Setting up Local Inference...

REM Check for Python
set PYTHON_CMD=python
where python >nul 2>nul
if errorlevel 1 (
    set PYTHON_CMD=py
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python is not installed
        echo Please install Python 3.8+ from: https://python.org
        pause
        exit /b 1
    )
)

REM Install Python dependencies
if exist requirements.txt (
    echo [INFO] Installing Python dependencies...
    %PYTHON_CMD% -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Some Python dependencies may have failed
    )
)

REM Start Python backend if inference.py exists
if exist inference.py (
    echo [INFO] Starting Python inference backend...
    start /B cmd /c "%PYTHON_CMD% inference.py > logs\inference.log 2>&1"
    echo [OK] Inference backend started
) else (
    echo [WARNING] inference.py not found - will run in mock mode
)

set BACKEND=local
goto START_SERVER

:DEMO
echo.
echo [INFO] Starting in Demo Mode...
set BACKEND=demo
goto START_SERVER

:START_SERVER
echo.
echo ============================================================
echo  Starting JARVIS AI System...
echo ============================================================
echo.

REM Start the server
echo [INFO] Starting web server on http://localhost:3001...
start /B cmd /c "node server.js > logs\server.log 2>&1"

REM Wait for server to start
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo  JARVIS AI is now running!
echo ============================================================
echo.
echo  Web Interface:  http://localhost:3001
echo  API Endpoint:   http://localhost:3001/api/chat
echo  Backend:        %BACKEND%
echo.
echo Press Ctrl+C to stop the server, or close this window.
echo.
echo ============================================================
echo.

REM Keep the script running
echo [INFO] Server is running. Close this window to stop.
echo [INFO] Check logs\server.log for details.
echo.
pause

:END
echo.
echo [INFO] Exiting...
endlocal
