# Windows Installer Scripts

This directory contains easy-to-use Windows installer scripts that automate the entire Jarvis setup, training, and Ollama integration process.

## 📁 Available Scripts

### 1. `install_train_jarvis_windows.bat`

**Batch file** - Double-click to run!

- ✅ Works on all Windows versions (7, 8, 10, 11)
- ✅ No special permissions needed
- ✅ Simple, straightforward execution
- ⚠️ Basic error handling

**How to use:**
1. Navigate to the project folder in File Explorer
2. Double-click `install_train_jarvis_windows.bat`
3. Follow the prompts

### 2. `install_train_jarvis_windows.ps1`

**PowerShell script** - Better features and logging

- ✅ Rich colored output
- ✅ Better error messages
- ✅ More robust error handling
- ⚠️ May require execution policy change

**How to use:**
```powershell
# Right-click on folder → Open in Terminal (or PowerShell)
.\install_train_jarvis_windows.ps1
```

**If you get an execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_train_jarvis_windows.ps1
```

## 🎯 What These Scripts Do

Both scripts follow the same workflow:

### Step 1: Check Python Installation
- Verifies Python is installed and in PATH
- Shows Python version
- Exits with helpful message if not found

### Step 2: Install Python Dependencies
- Upgrades pip to latest version
- Installs core dependencies (FastAPI, uvicorn, requests, numpy, matplotlib)
- Installs PyTorch (CPU version for compatibility)
- Installs ML tools (transformers, datasets, accelerate, peft)
- Installs all remaining requirements from `requirements.txt`

### Step 3: Check Ollama
- Checks if Ollama is installed
- Guides you to download if not found
- Optionally continues without Ollama (training only)
- Starts Ollama server if not running

### Step 4: Choose Training Option

You'll be asked to select one of three options:

#### Option 1: Quick Train (~5 minutes)
- Trains a simple Jarvis model
- Uses basic conversational training data
- Fast and easy, great for testing
- Runs `train_jarvis.py`

#### Option 2: Lab Train (~30 minutes)
- Advanced training with quantum experiments
- Generates training data from live lab API
- Fine-tunes with LoRA
- Converts to GGUF format
- Installs as `ben-lab` model in Ollama
- Pipeline:
  1. Start `jarvis_api.py` (if not running)
  2. Run `generate_lab_training_data.py`
  3. Run `finetune_ben_lab.py`
  4. Clone llama.cpp (if needed)
  5. Convert LoRA to GGUF
  6. Create Ollama model

#### Option 3: Skip Training
- No training, just setup
- Downloads a base Ollama model
- You can train later

### Step 5: Training Execution
Executes your chosen training option with progress updates

### Step 6: Install to Ollama
- Creates Modelfile with proper configuration
- Installs trained model to Ollama
- Provides usage instructions

## 🔍 Detailed Comparison

| Feature | .bat (Batch) | .ps1 (PowerShell) |
|---------|-------------|-------------------|
| Double-click to run | ✅ Yes | ❌ No (terminal required) |
| Colored output | ⚠️ Limited | ✅ Full colors |
| Error messages | ⚠️ Basic | ✅ Detailed |
| Windows compatibility | ✅ All versions | ✅ Win 7+ |
| Requires admin | ❌ No | ❌ No |
| Execution policy | N/A | ⚠️ May need change |
| Progress indicators | ⚠️ Basic | ✅ Enhanced |
| Error recovery | ⚠️ Basic | ✅ Better |

## 🛠️ Prerequisites

Before running these scripts, make sure you have:

1. **Python 3.9+** installed
   - Download from [python.org](https://www.python.org/downloads/)
   - ⚠️ **CRITICAL**: Check "Add Python to PATH" during installation!

2. **Ollama** (optional but recommended)
   - Download from [ollama.ai/download](https://ollama.ai/download)
   - Required for running trained models
   - Can train without it (install Ollama later)

3. **Git** (optional, for llama.cpp)
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Only needed for Lab Train option
   - Script will guide you if missing

4. **Sufficient Disk Space**
   - Quick Train: ~2 GB
   - Lab Train: ~10 GB (includes model downloads)

5. **Internet Connection**
   - For downloading dependencies and models

## 📊 Training Time Estimates

| Option | Time | Disk Space | Description |
|--------|------|-----------|-------------|
| Quick Train | ~5 min | ~2 GB | Simple model, basic data |
| Lab Train | ~30 min | ~10 GB | Advanced, experiment-based |
| Skip Training | ~2 min | ~1 GB | Just base model |

*Times vary based on internet speed and CPU performance*

## 🐛 Common Issues

### Issue: "Python is not recognized"

**Cause:** Python not in PATH

**Fix:**
1. Reinstall Python from [python.org](https://www.python.org/downloads/)
2. During installation, CHECK the box: "Add Python to PATH"
3. Restart your terminal/computer
4. Run the script again

### Issue: "Execution policy" error (PowerShell only)

**Cause:** Windows security restriction

**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the script again.

### Issue: "Ollama not found"

**Cause:** Ollama not installed or not in PATH

**Fix:**
1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Install it
3. Restart your terminal
4. Choose "Continue without Ollama" to just train, or
5. Run the script again after installing

### Issue: Training fails with "Out of memory"

**Cause:** Insufficient RAM (need ~8GB free)

**Fix:**
- Close other applications
- Choose Quick Train instead of Lab Train
- Edit training scripts to reduce batch size
- Restart your computer to free memory

### Issue: "Git not found" (Lab Train only)

**Cause:** Git not installed

**Fix:**
1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Install it
3. Restart your terminal
4. Run the script again

Or: Skip to Option 1 (Quick Train)

### Issue: llama-cpp-python build fails (CMake error)

**Cause:** Missing Visual Studio Build Tools (C++ compiler)

**What happens:**
- The installer shows "Building wheel for llama-cpp-python... error"
- CMake errors about NMake Makefiles or missing C++ support
- **This is normal and expected on most Windows machines!**

**Fix:**
The installer handles this automatically:
1. First tries to install pre-built wheels from official repository
2. If that fails, attempts to build from source
3. If build fails, continues anyway (most features work without it)

**If you want to install it manually later:**
```powershell
# Option 1: Try pre-built wheel (easiest)
python -m pip install --only-binary=:all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Option 2: Install Build Tools first, then build
# Download "Build Tools for Visual Studio" from visualstudio.microsoft.com
# Select "Desktop development with C++" during installation
# Then run: python -m pip install llama-cpp-python
```

**Important:** You don't need llama-cpp-python for most features! The training and Ollama integration work fine without it.

### Issue: Script window closes immediately

**Cause:** Python or other critical error

**Fix:**
1. Run from Command Prompt or PowerShell instead of double-clicking
2. This way you can see the error message
3. Address the specific error shown

### Issue: Downloads are very slow

**Cause:** Large model files (PyTorch ~2GB, etc.)

**Fix:**
- Be patient (first run takes longest)
- Ensure stable internet connection
- The downloads are cached, so subsequent runs are faster

## 🎮 After Installation

Once the script completes successfully, you can:

### Chat with your trained model:
```powershell
ollama run ben-lab
```

### Use the lab integration:
```powershell
# Terminal 1
python jarvis_api.py

# Terminal 2
python chat_with_lab.py
```

### Start the web interface:
```powershell
python streamlit_app.py
```

Then visit http://localhost:8501 in your browser

### Run experiments directly:
```powershell
python experiments/discovery_suite.py
```

## 🔄 Re-running the Scripts

You can run these scripts multiple times:

- If training failed: Run again to retry
- To train a different model: Choose different option
- To update dependencies: Run to reinstall
- Ollama integration failed: Run again after fixing Ollama

The scripts are **idempotent** (safe to run multiple times).

## 📚 Additional Documentation

- **Windows Quick Start**: `WINDOWS_QUICK_START.md` - Detailed Windows guide
- **Main Setup Guide**: `SETUP_JARVIS_LAB_OLLAMA.md` - Full lab setup
- **Quick Start**: `QUICK_START_BEN_LAB_LORA.md` - LoRA training quick ref
- **Main README**: `README.md` - Project overview

## 💡 Tips

1. **Run from project root** - Make sure you're in the main project directory
2. **Use Windows Terminal** - Better than Command Prompt
3. **PowerShell for better experience** - If you're comfortable with terminals
4. **Batch for simplicity** - If you want just double-click and go
5. **Virtual environment** - Consider using venv for Python packages:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
6. **Read error messages** - They often contain the solution
7. **Close antivirus temporarily** - Some AVs block script execution
8. **Run as non-admin first** - Only elevate if specifically needed

## 🎉 Success Indicators

You'll know the installation succeeded when:

- ✅ Script completes without errors
- ✅ "Installation Complete!" message appears
- ✅ `ollama list` shows your model (if not skipped)
- ✅ `python jarvis_api.py` starts without errors
- ✅ You can chat with the model

## 🔗 Getting Help

If you encounter issues:

1. Read this document thoroughly
2. Check `WINDOWS_QUICK_START.md`
3. Review error messages carefully
4. Search for the specific error online
5. Create a GitHub issue with:
   - Your Windows version
   - Python version (`python --version`)
   - Complete error message
   - Steps you've already tried

---

**Happy training! 🚀 Your local Jarvis awaits!**
