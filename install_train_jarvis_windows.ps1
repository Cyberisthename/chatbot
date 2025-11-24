# ========================================
# Jarvis Windows Easy Installer & Trainer
# PowerShell Version
# ========================================
# This script will:
# 1. Check Python installation
# 2. Install Python dependencies
# 3. Check/guide Ollama installation
# 4. Train Jarvis model
# 5. Install to Ollama
# ========================================

# Set up colors
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Cyan"
Clear-Host

# Set script directory as working directory
Set-Location $PSScriptRoot

# Header
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "    Jarvis Windows Easy Installer" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will install, train, and set up" -ForegroundColor White
Write-Host "Jarvis on your Windows machine!" -ForegroundColor White
Write-Host ""
Pause

# ========================================
# Step 1: Check Python
# ========================================
Write-Host ""
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
Write-Host ""

try {
    $pythonVersion = python --version 2>&1
    Write-Host $pythonVersion -ForegroundColor Green
    Write-Host "OK - Python is installed!" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Pause
    exit 1
}

# ========================================
# Step 2: Install Python Dependencies
# ========================================
Write-Host ""
Write-Host "[2/6] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor White
Write-Host ""

# Upgrade pip
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Could not upgrade pip" -ForegroundColor Yellow
}

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor White
python -m pip install fastapi uvicorn requests numpy matplotlib --quiet

# Try to install PyTorch (CPU version for Windows)
Write-Host "Installing PyTorch (this is a large download)..." -ForegroundColor White
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

# Install transformers and ML tools
Write-Host "Installing ML tools..." -ForegroundColor White
python -m pip install transformers datasets accelerate peft sentencepiece protobuf --quiet

# Install remaining dependencies from requirements.txt
Write-Host "Installing remaining dependencies..." -ForegroundColor White
if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt --quiet
}

Write-Host ""
Write-Host "OK - Dependencies installed!" -ForegroundColor Green

# ========================================
# Step 3: Check Ollama
# ========================================
Write-Host ""
Write-Host "[3/6] Checking Ollama installation..." -ForegroundColor Yellow
Write-Host ""

$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue
$skipOllama = $false

if (-not $ollamaInstalled) {
    Write-Host ""
    Write-Host "WARNING: Ollama is not installed!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Ollama is required to run the trained model." -ForegroundColor White
    Write-Host "Please download and install from: https://ollama.ai/download" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "After installing Ollama, you can:" -ForegroundColor White
    Write-Host "1. Run this script again, OR" -ForegroundColor White
    Write-Host "2. Skip Ollama setup for now and just train the model" -ForegroundColor White
    Write-Host ""
    
    $continue = Read-Host "Do you want to continue WITHOUT Ollama for now? (Y/N)"
    if ($continue -ne "Y" -and $continue -ne "y") {
        exit 1
    }
    $skipOllama = $true
}
else {
    $ollamaVersion = ollama --version 2>&1
    Write-Host $ollamaVersion -ForegroundColor Green
    Write-Host "OK - Ollama is installed!" -ForegroundColor Green
    
    # Check if Ollama is running
    Write-Host ""
    Write-Host "Checking if Ollama server is running..." -ForegroundColor White
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/version" -Method Get -ErrorAction Stop -TimeoutSec 2
        Write-Host "Ollama server is running!" -ForegroundColor Green
    }
    catch {
        Write-Host "Ollama is not running. Starting Ollama server..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Write-Host "Waiting for Ollama to start..." -ForegroundColor White
        Start-Sleep -Seconds 5
    }
}

# ========================================
# Step 4: Choose Training Path
# ========================================
Write-Host ""
Write-Host "[4/6] Choose training option:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Quick Train (Simple Jarvis model, ~5 minutes)" -ForegroundColor White
Write-Host "2. Lab Train (Advanced with experiments, ~30 minutes)" -ForegroundColor White
Write-Host "3. Skip training (use default Ollama models)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Select option (1-3)"

switch ($choice) {
    "1" { 
        # Quick Training
        Write-Host ""
        Write-Host "[5/6] Starting Quick Training..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "This will train a simple Jarvis model on basic data." -ForegroundColor White
        Write-Host ""
        
        python train_jarvis.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "ERROR: Training failed!" -ForegroundColor Red
            Write-Host "Check the error messages above." -ForegroundColor Yellow
            Pause
            exit 1
        }
        
        Write-Host ""
        Write-Host "OK - Training complete!" -ForegroundColor Green
        $trainedModel = "quick"
    }
    
    "2" {
        # Lab Training
        Write-Host ""
        Write-Host "[5/6] Starting Lab Training Pipeline..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "This will:" -ForegroundColor White
        Write-Host "1. Start Jarvis Lab API (if not running)" -ForegroundColor White
        Write-Host "2. Generate training data from experiments" -ForegroundColor White
        Write-Host "3. Fine-tune model with LoRA" -ForegroundColor White
        Write-Host "4. Convert to Ollama format" -ForegroundColor White
        Write-Host ""
        
        # Check if lab API is running
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -Method Get -ErrorAction Stop -TimeoutSec 2
            Write-Host "Jarvis Lab API is already running!" -ForegroundColor Green
        }
        catch {
            Write-Host "Starting Jarvis Lab API..." -ForegroundColor Yellow
            Start-Process -FilePath "python" -ArgumentList "jarvis_api.py" -WindowStyle Normal
            Write-Host "Waiting for API to start..." -ForegroundColor White
            Start-Sleep -Seconds 10
        }
        
        Write-Host ""
        Write-Host "Generating training data from experiments..." -ForegroundColor White
        python generate_lab_training_data.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "ERROR: Failed to generate training data!" -ForegroundColor Red
            Write-Host "Make sure jarvis_api.py is running." -ForegroundColor Yellow
            Pause
            exit 1
        }
        
        Write-Host ""
        Write-Host "Fine-tuning model with LoRA..." -ForegroundColor White
        python finetune_ben_lab.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "ERROR: Fine-tuning failed!" -ForegroundColor Red
            Pause
            exit 1
        }
        
        Write-Host ""
        Write-Host "Converting to GGUF format..." -ForegroundColor White
        
        # Clone llama.cpp if needed
        if (-not (Test-Path "llama.cpp")) {
            Write-Host "Downloading llama.cpp..." -ForegroundColor White
            git clone https://github.com/ggerganov/llama.cpp.git
            if ($LASTEXITCODE -ne 0) {
                Write-Host ""
                Write-Host "ERROR: Failed to clone llama.cpp" -ForegroundColor Red
                Write-Host "You need Git installed: https://git-scm.com/download/win" -ForegroundColor Yellow
                Pause
                exit 1
            }
        }
        
        python llama.cpp/scripts/convert_lora_to_gguf.py --adapter-dir ben-lab-lora --outfile ben-lab-adapter.gguf
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "ERROR: Conversion to GGUF failed!" -ForegroundColor Red
            Pause
            exit 1
        }
        
        Write-Host ""
        Write-Host "OK - Lab training complete!" -ForegroundColor Green
        $trainedModel = "lab"
    }
    
    "3" {
        # Skip Training
        Write-Host ""
        Write-Host "[5/6] Skipping training..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "You can use default Ollama models." -ForegroundColor White
        Write-Host "Downloading a base model..." -ForegroundColor White
        Write-Host ""
        
        if ($skipOllama) {
            Write-Host "Cannot download models - Ollama is not installed." -ForegroundColor Yellow
        }
        else {
            ollama pull llama3.2:1b
            if ($LASTEXITCODE -ne 0) {
                Write-Host ""
                Write-Host "Warning: Could not pull model. You can do this later:" -ForegroundColor Yellow
                Write-Host "  ollama pull llama3.2:1b" -ForegroundColor White
            }
        }
        
        $trainedModel = "skip"
    }
    
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

# ========================================
# Install to Ollama
# ========================================
if ($trainedModel -ne "skip") {
    Write-Host ""
    Write-Host "[6/6] Installing to Ollama..." -ForegroundColor Yellow
    Write-Host ""
    
    if ($skipOllama) {
        Write-Host ""
        Write-Host "Ollama is not installed. Skipping installation." -ForegroundColor Yellow
        Write-Host "Your trained model is saved but not installed to Ollama." -ForegroundColor White
        Write-Host ""
        Write-Host "Install Ollama from https://ollama.ai/download" -ForegroundColor Cyan
        Write-Host "Then run: ollama create jarvis -f Modelfile" -ForegroundColor White
    }
    elseif ($trainedModel -eq "lab") {
        # Create Modelfile for lab-trained model
        Write-Host "Creating Modelfile..." -ForegroundColor White
        
        $modelfileContent = @"
FROM llama3.2:1b

ADAPTER ./ben-lab-adapter.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM """
You are Ben's Lab AI (Jarvis-2v).
You understand the Jarvis-2v quantum phase simulator, TRI, replay drift, clustering,
and the lab API. You explain results clearly and use Ben's terminology.
"""
"@
        
        Set-Content -Path "Modelfile" -Value $modelfileContent
        
        Write-Host "Installing ben-lab model to Ollama..." -ForegroundColor White
        ollama create ben-lab -f Modelfile
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "ERROR: Failed to create Ollama model!" -ForegroundColor Red
            Pause
            exit 1
        }
        
        Write-Host ""
        Write-Host "SUCCESS! Model installed as 'ben-lab'" -ForegroundColor Green
        Write-Host ""
        Write-Host "To use it, run:" -ForegroundColor White
        Write-Host "  ollama run ben-lab" -ForegroundColor Cyan
        Write-Host ""
    }
    elseif ($trainedModel -eq "quick") {
        Write-Host ""
        Write-Host "For quick-trained models, you'll need to:" -ForegroundColor White
        Write-Host "1. Convert the model to GGUF format" -ForegroundColor White
        Write-Host "2. Create a Modelfile" -ForegroundColor White
        Write-Host "3. Run: ollama create jarvis -f Modelfile" -ForegroundColor White
        Write-Host ""
        Write-Host "See the documentation for details." -ForegroundColor White
    }
}

# ========================================
# Done
# ========================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "     Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "What you can do next:" -ForegroundColor Yellow
Write-Host ""

if (-not $skipOllama) {
    if (Test-Path "ben-lab-adapter.gguf") {
        Write-Host "  ollama run ben-lab          - Chat with your trained model" -ForegroundColor Cyan
    }
    else {
        Write-Host "  ollama run llama3.2:1b      - Chat with base model" -ForegroundColor Cyan
    }
    Write-Host "  python chat_with_lab.py      - Use lab integration" -ForegroundColor Cyan
}

Write-Host "  python jarvis_api.py          - Start lab API" -ForegroundColor Cyan
Write-Host "  python streamlit_app.py       - Web interface" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check the README.md for more information!" -ForegroundColor White
Write-Host ""
Pause
