# ==============================================================================
# JARVIS AI - PowerShell Start Script for Windows
# ==============================================================================

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Print banner
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "     J  A  R  V  I  S    A  I    S  Y  S  T  E  M       " -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "              Easy Run Script v1.0 (PowerShell)         " -ForegroundColor Cyan
Write-Host "              Ollama | Pinokio | Local Inference         " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Function to check if command exists
function Test-Command {
    param($Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $Command) { return $true }
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Check for Node.js
Write-Host "[INFO] Checking Node.js..." -ForegroundColor Blue
if (Test-Command "node") {
    $nodeVersion = node -v
    Write-Host "[OK] Node.js is installed: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Node.js is not installed" -ForegroundColor Red
    Write-Host "Please install Node.js from: https://nodejs.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install Node dependencies if needed
if (-not (Test-Path "node_modules")) {
    Write-Host "[INFO] Installing Node dependencies..." -ForegroundColor Blue
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check backend availability
$BackendAvailable = $false

# Check Ollama
Write-Host "[INFO] Checking Ollama..." -ForegroundColor Blue
if (Test-Command "ollama") {
    $BackendAvailable = $true
    Write-Host "[OK] Ollama detected" -ForegroundColor Green
} else {
    Write-Host "[INFO] Ollama not found" -ForegroundColor Gray
}

# Check Python
Write-Host "[INFO] Checking Python..." -ForegroundColor Blue
if (Test-Command "python") {
    $BackendAvailable = $true
    $pythonVersion = python --version
    Write-Host "[OK] Python detected: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[INFO] Python not found" -ForegroundColor Gray
}

if (-not $BackendAvailable) {
    Write-Host "[WARNING] No AI backend detected" -ForegroundColor Yellow
    Write-Host "[INFO] You can still run in demo mode" -ForegroundColor Gray
}

# Show menu
Write-Host ""
Write-Host "Available Options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Start with Ollama (if installed)" -ForegroundColor White
Write-Host "  2. Start with Local Inference (if Python available)" -ForegroundColor White
Write-Host "  3. Start in Demo Mode (no AI backend)" -ForegroundColor White
Write-Host "  4. Exit" -ForegroundColor White
Write-Host ""

$Choice = Read-Host "Select option (1-4)"

switch ($Choice) {
    "1" {
        Write-Host ""
        Write-Host "[INFO] Setting up Ollama..." -ForegroundColor Blue
        if (-not (Test-Command "ollama")) {
            Write-Host "[ERROR] Ollama is not installed" -ForegroundColor Red
            Write-Host "Please install from: https://ollama.ai/download" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
        
        Write-Host "[INFO] Checking Ollama service..." -ForegroundColor Blue
        $ollamaList = ollama list 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARNING] Ollama service may not be running" -ForegroundColor Yellow
            Write-Host "Try running: ollama serve" -ForegroundColor Gray
            Write-Host ""
        }
        
        $ModelName = Read-Host "Enter model name (default: llama3.2)"
        if ([string]::IsNullOrWhiteSpace($ModelName)) {
            $ModelName = "llama3.2"
        }
        
        Write-Host "[INFO] Using model: $ModelName" -ForegroundColor Blue
        $Backend = "ollama"
    }
    "2" {
        Write-Host ""
        Write-Host "[INFO] Setting up Local Inference..." -ForegroundColor Blue
        
        if (-not (Test-Command "python")) {
            Write-Host "[ERROR] Python is not installed" -ForegroundColor Red
            Write-Host "Please install Python 3.8+ from: https://python.org" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
        
        # Install Python dependencies
        if (Test-Path "requirements.txt") {
            Write-Host "[INFO] Installing Python dependencies..." -ForegroundColor Blue
            python -m pip install -q -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARNING] Some Python dependencies may have failed" -ForegroundColor Yellow
            }
        }
        
        # Start Python backend
        if (Test-Path "inference.py") {
            Write-Host "[INFO] Starting Python inference backend..." -ForegroundColor Blue
            Start-Process -FilePath "python" -ArgumentList "inference.py" -RedirectStandardOutput "logs\inference.log" -RedirectStandardError "logs\inference.log" -WindowStyle Hidden
            Write-Host "[OK] Inference backend started" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] inference.py not found - will run in mock mode" -ForegroundColor Yellow
        }
        
        $Backend = "local"
    }
    "3" {
        Write-Host ""
        Write-Host "[INFO] Starting in Demo Mode..." -ForegroundColor Blue
        $Backend = "demo"
    }
    "4" {
        Write-Host ""
        Write-Host "[INFO] Exiting..." -ForegroundColor Gray
        exit 0
    }
    default {
        Write-Host "[ERROR] Invalid choice" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Start server
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Starting JARVIS AI System..." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[INFO] Starting web server on http://localhost:3001..." -ForegroundColor Blue
$ProcessInfo = Start-Process -FilePath "node" -ArgumentList "server.js" -RedirectStandardOutput "logs\server.log" -RedirectStandardError "logs\server.log" -PassThru

# Wait for server to start
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  JARVIS AI is now running!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Web Interface:  http://localhost:3001" -ForegroundColor Cyan
Write-Host "  API Endpoint:   http://localhost:3001/api/chat" -ForegroundColor Cyan
Write-Host "  Backend:        $Backend" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server, or close this window." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Keep the script running
Write-Host "[INFO] Server is running. Close this window to stop." -ForegroundColor Blue
Write-Host "[INFO] Check logs\server.log for details." -ForegroundColor Blue
Write-Host ""

# Wait for user to press Ctrl+C
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host ""
    Write-Host "[INFO] Stopping server..." -ForegroundColor Blue
    if ($ProcessInfo -and -not $ProcessInfo.HasExited) {
        Stop-Process -Id $ProcessInfo.Id -Force
    }
    Write-Host "[OK] Server stopped" -ForegroundColor Green
}
