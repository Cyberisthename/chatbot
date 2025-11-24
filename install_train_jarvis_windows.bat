@echo off
REM ========================================
REM Jarvis Windows Easy Installer & Trainer
REM ========================================
REM This script will:
REM 1. Check Python installation
REM 2. Install Python dependencies
REM 3. Check/guide Ollama installation
REM 4. Train Jarvis model
REM 5. Install to Ollama
REM ========================================

SETLOCAL ENABLEDELAYEDEXPANSION
COLOR 0B
TITLE Jarvis Easy Installer

echo.
echo ============================================
echo     Jarvis Windows Easy Installer
echo ============================================
echo.
echo This script will install, train, and set up
echo Jarvis on your Windows machine!
echo.
pause

REM Change to script directory
cd /d "%~dp0"

REM ========================================
REM Step 1: Check Python
REM ========================================
echo.
echo [1/6] Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

python --version
echo OK - Python is installed!

REM ========================================
REM Step 2: Install Python Dependencies
REM ========================================
echo.
echo [2/6] Installing Python dependencies...
echo This may take several minutes...
echo.

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Warning: Could not upgrade pip
)

REM Install core dependencies
python -m pip install fastapi uvicorn requests numpy matplotlib

REM Try to install PyTorch (CPU version for Windows)
echo.
echo Installing PyTorch (this is a large download)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install transformers and ML tools
python -m pip install transformers datasets accelerate peft sentencepiece protobuf

REM Install remaining dependencies
python -m pip install -r requirements.txt

echo.
echo OK - Dependencies installed!

REM ========================================
REM Step 3: Check Ollama
REM ========================================
echo.
echo [3/6] Checking Ollama installation...
echo.

where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama is not installed!
    echo.
    echo Ollama is required to run the trained model.
    echo Please download and install from: https://ollama.ai/download
    echo.
    echo After installing Ollama, you can:
    echo 1. Run this script again, OR
    echo 2. Skip Ollama setup for now and just train the model
    echo.
    choice /C YN /M "Do you want to continue WITHOUT Ollama for now"
    if errorlevel 2 exit /b 1
    set SKIP_OLLAMA=1
) else (
    ollama --version
    echo OK - Ollama is installed!
    set SKIP_OLLAMA=0
    
    REM Check if Ollama is running
    echo.
    echo Checking if Ollama server is running...
    curl -s http://localhost:11434/api/version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Ollama is not running. Starting Ollama server...
        start "Ollama Server" ollama serve
        echo Waiting for Ollama to start...
        timeout /t 5 /nobreak >nul
    )
)

REM ========================================
REM Step 4: Choose Training Path
REM ========================================
echo.
echo [4/6] Choose training option:
echo.
echo 1. Quick Train (Simple Jarvis model, ~5 minutes)
echo 2. Lab Train (Advanced with experiments, ~30 minutes)
echo 3. Skip training (use default Ollama models)
echo.
choice /C 123 /M "Select option"

if errorlevel 3 goto :skip_training
if errorlevel 2 goto :lab_training
if errorlevel 1 goto :quick_training

REM ========================================
REM Quick Training
REM ========================================
:quick_training
echo.
echo [5/6] Starting Quick Training...
echo.
echo This will train a simple Jarvis model on basic data.
echo.

python train_jarvis.py
if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo OK - Training complete!
goto :install_to_ollama

REM ========================================
REM Lab Training
REM ========================================
:lab_training
echo.
echo [5/6] Starting Lab Training Pipeline...
echo.
echo This will:
echo 1. Start Jarvis Lab API (if not running)
echo 2. Generate training data from experiments
echo 3. Fine-tune model with LoRA
echo 4. Convert to Ollama format
echo.

REM Check if lab API is running
curl -s http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    echo.
    echo Starting Jarvis Lab API...
    start "Jarvis Lab API" python jarvis_api.py
    echo Waiting for API to start...
    timeout /t 10 /nobreak >nul
)

echo.
echo Generating training data from experiments...
python generate_lab_training_data.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate training data!
    echo Make sure jarvis_api.py is running.
    pause
    exit /b 1
)

echo.
echo Fine-tuning model with LoRA...
python finetune_ben_lab.py
if errorlevel 1 (
    echo.
    echo ERROR: Fine-tuning failed!
    pause
    exit /b 1
)

echo.
echo Converting to GGUF format...

REM Clone llama.cpp if needed
if not exist "llama.cpp" (
    echo Downloading llama.cpp...
    git clone https://github.com/ggerganov/llama.cpp.git
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to clone llama.cpp
        echo You need Git installed: https://git-scm.com/download/win
        pause
        exit /b 1
    )
)

python llama.cpp/scripts/convert_lora_to_gguf.py --adapter-dir ben-lab-lora --outfile ben-lab-adapter.gguf
if errorlevel 1 (
    echo.
    echo ERROR: Conversion to GGUF failed!
    pause
    exit /b 1
)

echo.
echo OK - Lab training complete!
goto :install_to_ollama

REM ========================================
REM Skip Training
REM ========================================
:skip_training
echo.
echo [5/6] Skipping training...
echo.
echo You can use default Ollama models.
echo Downloading a base model...
echo.

if !SKIP_OLLAMA!==1 (
    echo Cannot download models - Ollama is not installed.
    goto :done
)

ollama pull llama3.2:1b
if errorlevel 1 (
    echo.
    echo Warning: Could not pull model. You can do this later:
    echo   ollama pull llama3.2:1b
)

goto :done

REM ========================================
REM Install to Ollama
REM ========================================
:install_to_ollama
echo.
echo [6/6] Installing to Ollama...
echo.

if !SKIP_OLLAMA!==1 (
    echo.
    echo Ollama is not installed. Skipping installation.
    echo Your trained model is saved but not installed to Ollama.
    echo.
    echo Install Ollama from https://ollama.ai/download
    echo Then run: ollama create jarvis -f Modelfile
    goto :done
)

REM Create Modelfile
echo Creating Modelfile...

if exist "ben-lab-adapter.gguf" (
    REM Lab-trained model
    (
        echo FROM llama3.2:1b
        echo.
        echo ADAPTER ./ben-lab-adapter.gguf
        echo.
        echo PARAMETER temperature 0.2
        echo PARAMETER top_p 0.9
        echo.
        echo SYSTEM """
        echo You are Ben's Lab AI ^(Jarvis-2v^).
        echo You understand the Jarvis-2v quantum phase simulator, TRI, replay drift, clustering,
        echo and the lab API. You explain results clearly and use Ben's terminology.
        echo """
    ) > Modelfile
    
    echo Installing ben-lab model to Ollama...
    ollama create ben-lab -f Modelfile
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create Ollama model!
        pause
        exit /b 1
    )
    
    echo.
    echo SUCCESS! Model installed as 'ben-lab'
    echo.
    echo To use it, run:
    echo   ollama run ben-lab
    echo.
) else if exist "jarvis-model" (
    REM Quick-trained model
    echo.
    echo For quick-trained models, you'll need to:
    echo 1. Convert the model to GGUF format
    echo 2. Create a Modelfile
    echo 3. Run: ollama create jarvis -f Modelfile
    echo.
    echo See the documentation for details.
) else (
    echo.
    echo No trained model found to install.
)

goto :done

REM ========================================
REM Done
REM ========================================
:done
echo.
echo ============================================
echo     Installation Complete!
echo ============================================
echo.
echo What you can do next:
echo.

if !SKIP_OLLAMA!==0 (
    if exist "ben-lab-adapter.gguf" (
        echo   ollama run ben-lab          - Chat with your trained model
    ) else (
        echo   ollama run llama3.2:1b      - Chat with base model
    )
    echo   python chat_with_lab.py      - Use lab integration
)

echo   python jarvis_api.py          - Start lab API
echo   python streamlit_app.py       - Web interface
echo.
echo Check the README.md for more information!
echo.
pause
