# 📦 JARVIS AI - Complete Installation Guide

This guide will walk you through installing JARVIS AI on your system.

## 🎯 Quick Install (Recommended)

### Linux / Mac / WSL

```bash
# 1. Clone or download the project
cd /path/to/jarvis-ai

# 2. Run the auto-installer
./install_prerequisites.sh

# 3. Run JARVIS
./run_ai.sh
```

### Windows

```powershell
# 1. Download and extract the project
# 2. Double-click install_prerequisites.bat (if available)
#    Or manually install dependencies:
#    - Node.js from https://nodejs.org
#    - Python from https://python.org
#    - Ollama from https://ollama.ai/download (optional)

# 3. Run JARVIS
# Double-click: run_ai.bat
# Or PowerShell: .\Start-JARVIS.ps1
```

## 📋 System Requirements

### Minimum Requirements
- **CPU**: Any modern multi-core processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended for AI Inference
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- **RAM**: 16GB+ for larger models
- **Storage**: SSD preferred

### Dependencies
- **Node.js** 16+ (Required for web UI)
- **Python** 3.8+ (Required for local inference)
- **Ollama** (Optional - for easy model management)
- **Git** (Optional - for cloning the repo)

## 🔧 Detailed Installation

### Step 1: Install Node.js

Node.js is required for the web interface.

#### Windows
1. Download from https://nodejs.org
2. Run the installer (LTS version recommended)
3. Restart your terminal/command prompt

#### macOS
```bash
# Using Homebrew
brew install node

# Or download from https://nodejs.org
```

#### Linux (Debian/Ubuntu)
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### Linux (Arch/Manjaro)
```bash
sudo pacman -S nodejs npm
```

Verify installation:
```bash
node --version
npm --version
```

### Step 2: Install Python 3

Python is required for local inference.

#### Windows
1. Download from https://python.org
2. Run the installer
3. ⚠️ **Important**: Check "Add Python to PATH"
4. Restart your terminal

#### macOS
```bash
# Using Homebrew
brew install python3

# Or download from https://python.org
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

#### Linux (Arch/Manjaro)
```bash
sudo pacman -S python python-pip
```

Verify installation:
```bash
python3 --version
pip3 --version
```

### Step 3: Install Ollama (Optional but Recommended)

Ollama makes it easy to download and run AI models.

#### Windows
1. Download from https://ollama.ai/download
2. Run the installer
3. Open a new terminal

#### macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

### Step 4: Install Project Dependencies

#### For Local Inference (Python)
```bash
pip3 install -r requirements.txt
```

#### For Web UI (Node.js)
```bash
npm install
```

### Step 5: Download or Train a Model

#### Option A: Use Ollama (Easiest)
```bash
# Pull a model
ollama pull llama3.2

# List available models
ollama list
```

#### Option B: Use Your Trained Model
```bash
# Train your own model
python train_jarvis.py

# Export to GGUF
python train_and_export_gguf.py

# Model will be in models/ directory
```

#### Option C: Download Pre-trained GGUF
1. Visit Hugging Face
2. Search for GGUF models
3. Download to `models/` directory

### Step 6: Run JARVIS

```bash
# Linux/Mac
./run_ai.sh

# Windows
run_ai.bat
# Or
.\Start-JARVIS.ps1
```

## 🎯 Installation by Use Case

### Use Case 1: Quick Demo (No Model)
```bash
# Just run the script and select "Demo" mode
./run_ai.sh
```

### Use Case 2: Ollama + Web UI (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Run JARVIS
./run_ai.sh
# Select "Ollama" from menu
```

### Use Case 3: Custom Trained Model
```bash
# Train model on your data
python train_jarvis.py --data-path ./my_data/

# Export to GGUF
python train_and_export_gguf.py

# Run with local inference
./run_ai.sh
# Select "Local" from menu
```

### Use Case 4: Development Setup
```bash
# Install all dependencies
./install_prerequisites.sh

# Install development tools
npm install --save-dev nodemon

# Run in development mode
npm run dev
```

## 🔍 Verification Steps

After installation, verify everything works:

### 1. Check Node.js
```bash
node --version
# Should show: v16.x.x or higher
```

### 2. Check Python
```bash
python3 --version
# Should show: Python 3.8.x or higher
```

### 3. Check Dependencies
```bash
# Node modules exist
ls node_modules

# Python packages installed
pip3 list
```

### 4. Check Ollama (if installed)
```bash
ollama --version
ollama list
```

### 5. Run Server
```bash
./run_ai.sh
# Should start without errors
```

### 6. Open Web UI
- Navigate to http://localhost:3001
- You should see the JARVIS interface

## 🐛 Troubleshooting

### Node.js Issues

**"node: command not found"**
- Node.js not installed or not in PATH
- Reinstall Node.js and ensure PATH is set correctly

**"npm install fails"**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and package-lock.json
- Run `npm install` again

### Python Issues

**"python3: command not found"**
- Python not installed or not in PATH
- On Windows, try `python` instead of `python3`

**"pip not found"**
- Python installation may have been incomplete
- Reinstall Python and ensure "Add to PATH" is checked

**"Module not found" errors**
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Ollama Issues

**"ollama: command not found"**
- Ollama not installed
- Run: `curl -fsSL https://ollama.ai/install.sh | sh`

**"Failed to connect to Ollama"**
- Start Ollama service: `ollama serve`
- Check if Ollama is running in another terminal

### Port Already in Use

**"Port 3001 is already in use"**
```bash
# Find and kill the process
# Linux/Mac:
lsof -ti:3001 | xargs kill -9

# Windows:
netstat -ano | findstr :3001
taskkill /PID <PID> /F

# Or use a different port
PORT=3002 ./run_ai.sh
```

### Permission Issues

**"Permission denied" on Linux/Mac**
```bash
# Make scripts executable
chmod +x run_ai.sh
chmod +x install_prerequisites.sh
```

**"Access denied" on Windows**
- Run PowerShell as Administrator
- Or run Command Prompt as Administrator

### Model Loading Issues

**"Model not found"**
- Ensure model file is in `models/` directory
- Verify model path in `config.yaml`
- Check file format is GGUF

**"Out of memory"**
- Use a smaller quantized model
- Reduce `context_size` in config.yaml
- Close other applications

## 🔄 Updating JARVIS

### Using Git
```bash
git pull origin main
npm install
pip3 install -r requirements.txt
```

### Manual Update
1. Download new version
2. Copy `models/` directory to new version
3. Run install scripts
4. Start with `./run_ai.sh`

## 🗑️ Uninstallation

### Remove JARVIS Files
```bash
# Remove project directory
rm -rf /path/to/jarvis-ai

# Or on Windows, delete the folder
```

### Remove Dependencies (Optional)

**Remove Node.js:**
- Windows: Use "Add or Remove Programs"
- macOS: `brew uninstall node`
- Linux: `sudo apt remove nodejs npm`

**Remove Python:**
- Windows: Use "Add or Remove Programs"
- macOS: `brew uninstall python3`
- Linux: `sudo apt remove python3 python3-pip`

**Remove Ollama:**
```bash
# Linux/Mac
ollama uninstall --all
# Then remove Ollama app from Applications

# Windows
# Use "Add or Remove Programs"
```

## 📚 Additional Resources

- [Quick Start Guide](README_EASY_RUN.md)
- [Detailed Local AI Guide](QUICKSTART_LOCAL_AI.md)
- [Main README](README.md)
- [Vercel Deployment](VERCEL_DEPLOYMENT.md)
- [API Documentation](docs/api.md)

## 🤝 Getting Help

If you encounter issues:

1. **Check the logs**: `logs/server.log` and `logs/inference.log`
2. **Verify dependencies**: Run through verification steps
3. **Check error messages**: Read error messages carefully
4. **Search online**: Many common issues have solutions online
5. **Ask for help**: Provide system details and error messages

## ✅ Installation Checklist

- [ ] Node.js installed (v16+)
- [ ] Python 3 installed (v3.8+)
- [ ] Ollama installed (optional)
- [ ] Node dependencies installed (`npm install`)
- [ ] Python dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Model downloaded or trained
- [ ] Run script is executable (`chmod +x run_ai.sh`)
- [ ] Can start JARVIS (`./run_ai.sh`)
- [ ] Web UI loads (http://localhost:3001)
- [ ] Can chat with AI

## 🎉 Installation Complete!

You're ready to use JARVIS AI! Start with:

```bash
./run_ai.sh
```

And open http://localhost:3001 in your browser.

---

**Need help? Check the logs in the `logs/` directory for detailed error messages.**
