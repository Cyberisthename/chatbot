# 📖 Manual Installation Guide

**Complete step-by-step manual installation for worst-case scenarios**

If the automated scripts don't work, follow this detailed manual guide to install Jarvis Quantum LLM into Ollama.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Method 1: Standard Installation](#method-1-standard-installation)
3. [Method 2: Manual File Placement](#method-2-manual-file-placement)
4. [Method 3: Direct Ollama Directory](#method-3-direct-ollama-directory)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Ollama** (any recent version)
   - Linux/Mac: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Windows: Download from https://ollama.ai/download
   - Verify: `ollama --version`

2. **Python 3.7+**
   - Check: `python3 --version`
   - If missing, install from https://www.python.org/downloads/

3. **NumPy**
   - Install: `pip3 install numpy`
   - Verify: `python3 -c "import numpy; print(numpy.__version__)"`

### Required Files

All files should be in the `ollama-jarvis-setup/` directory:

- `jarvis_quantum_llm.npz` (in `../ready-to-deploy-hf/`) - Model weights
- `config.json` (in `../ready-to-deploy-hf/`) - Model configuration
- `numpy_to_gguf.py` - Conversion script
- `Modelfile` - Ollama configuration

---

## Method 1: Standard Installation

**This is the recommended method. Try this first.**

### Step 1: Navigate to Setup Directory

```bash
cd ollama-jarvis-setup
```

### Step 2: Install Dependencies

```bash
pip3 install numpy requests
```

### Step 3: Convert Model to GGUF

```bash
python3 numpy_to_gguf.py
```

**Expected output:**
```
Loading NumPy model from ../ready-to-deploy-hf/jarvis_quantum_llm.npz...
Converting to GGUF format...
Writing GGUF file...
✅ Conversion complete: jarvis-quantum.gguf
```

**Verify the file was created:**
```bash
ls -lh jarvis-quantum.gguf
```
Should show a file ~45-50MB in size.

### Step 4: Create Ollama Model

```bash
ollama create jarvis -f Modelfile
```

**Expected output:**
```
transferring model data
using existing layer sha256:xxxxx
creating new layer sha256:xxxxx
writing manifest
success
```

### Step 5: Test the Model

```bash
ollama run jarvis
```

Type a test prompt:
```
>>> What is quantum mechanics?
```

If you get a response, success! 🎉

---

## Method 2: Manual File Placement

**If the standard method fails, try this approach.**

### Step 1: Find Your Ollama Directory

**Linux/Mac:**
```bash
echo $HOME/.ollama
ls -la ~/.ollama/models
```

**Windows (PowerShell):**
```powershell
echo $env:USERPROFILE\.ollama
dir $env:USERPROFILE\.ollama\models
```

**Or check environment variable:**
```bash
echo $OLLAMA_MODELS
```

### Step 2: Convert Model (if not done)

```bash
cd ollama-jarvis-setup
python3 numpy_to_gguf.py
```

### Step 3: Copy GGUF File

**Linux/Mac:**
```bash
mkdir -p ~/.ollama/models/blobs
cp jarvis-quantum.gguf ~/.ollama/models/blobs/
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ollama\models\blobs"
Copy-Item jarvis-quantum.gguf "$env:USERPROFILE\.ollama\models\blobs\"
```

### Step 4: Create Model with Absolute Path

Edit `Modelfile` to use absolute path:

**Linux/Mac:**
```bash
# Create temporary Modelfile with absolute path
cp Modelfile Modelfile.tmp
sed -i "1s|FROM ./jarvis-quantum.gguf|FROM $HOME/.ollama/models/blobs/jarvis-quantum.gguf|" Modelfile.tmp
ollama create jarvis -f Modelfile.tmp
rm Modelfile.tmp
```

**Windows (manually edit Modelfile):**
```
FROM C:\Users\YourUsername\.ollama\models\blobs\jarvis-quantum.gguf
```

Then run:
```powershell
ollama create jarvis -f Modelfile
```

### Step 5: Test

```bash
ollama run jarvis
```

---

## Method 3: Direct Ollama Directory

**Last resort: manually create all necessary files in Ollama's directory structure.**

### Understanding Ollama's Structure

Ollama stores models in this structure:
```
~/.ollama/models/
├── blobs/              # Model weights and data
│   └── sha256-xxxxx    # Hash-named files
└── manifests/          # Model manifests
    └── registry.ollama.ai/
        └── library/
            └── jarvis/
                └── latest
```

### Step 1: Convert and Get Hash

```bash
cd ollama-jarvis-setup
python3 numpy_to_gguf.py
sha256sum jarvis-quantum.gguf  # Linux/Mac
# or
certutil -hashfile jarvis-quantum.gguf SHA256  # Windows
```

Save this hash (example: `abc123...xyz789`)

### Step 2: Copy to Blobs

**Linux/Mac:**
```bash
OLLAMA_DIR="$HOME/.ollama/models"
HASH=$(sha256sum jarvis-quantum.gguf | cut -d' ' -f1)

# Copy with hash name
mkdir -p "$OLLAMA_DIR/blobs"
cp jarvis-quantum.gguf "$OLLAMA_DIR/blobs/sha256-$HASH"
```

**Windows:**
```powershell
$OLLAMA_DIR = "$env:USERPROFILE\.ollama\models"
$HASH = (Get-FileHash jarvis-quantum.gguf -Algorithm SHA256).Hash.ToLower()

# Copy with hash name
New-Item -ItemType Directory -Force -Path "$OLLAMA_DIR\blobs"
Copy-Item jarvis-quantum.gguf "$OLLAMA_DIR\blobs\sha256-$HASH"
```

### Step 3: Create Model via Ollama CLI

Even with manual placement, use `ollama create`:

```bash
ollama create jarvis -f Modelfile
```

This ensures Ollama properly indexes the model.

---

## Verification

### Check Model is Listed

```bash
ollama list
```

Should show:
```
NAME      ID            SIZE    MODIFIED
jarvis    xxxxx         45MB    X minutes ago
```

### Test Basic Functionality

```bash
echo "What is 2+2?" | ollama run jarvis
```

Should get a response (even if not perfectly accurate, proves it works).

### Check Model Info

```bash
ollama show jarvis
```

Shows model details, parameters, and system prompt.

---

## Troubleshooting

### Problem: "model not found"

**Solution 1:** Ensure Ollama is running
```bash
# Start Ollama server (if not running)
ollama serve
```

**Solution 2:** Recreate the model
```bash
ollama rm jarvis
ollama create jarvis -f Modelfile
```

**Solution 3:** Check model list
```bash
ollama list | grep jarvis
```

### Problem: "failed to load model"

**Cause:** GGUF file corrupted or incomplete

**Solution:**
1. Delete GGUF file: `rm jarvis-quantum.gguf`
2. Reconvert: `python3 numpy_to_gguf.py`
3. Verify size: `ls -lh jarvis-quantum.gguf` (should be ~45-50MB)
4. Recreate model: `ollama create jarvis -f Modelfile`

### Problem: "cannot connect to ollama server"

**Solution Linux/Mac:**
```bash
# Start server in background
ollama serve &

# Wait a moment
sleep 2

# Try again
ollama list
```

**Solution Windows:**
- Ensure Ollama desktop app is running
- Or run `ollama serve` in a separate terminal

### Problem: "Modelfile parse error"

**Solution:**
Check `Modelfile` syntax:
```bash
cat Modelfile
```

Ensure:
- `FROM` line points to valid file
- No extra quotes or spaces
- Parameters use correct syntax: `PARAMETER name value`

**Fix common issues:**
```bash
# Ensure GGUF exists in current directory
ls -l jarvis-quantum.gguf

# Use relative path
# Modelfile should have: FROM ./jarvis-quantum.gguf
```

### Problem: "conversion script fails"

**Solution:**
```bash
# Check NumPy is installed
python3 -c "import numpy; print(numpy.__version__)"

# If not installed
pip3 install --upgrade numpy

# Check source file exists
ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz

# Try conversion with verbose output
python3 -u numpy_to_gguf.py
```

### Problem: "Python not found"

**Solution Linux:**
```bash
# Install Python 3
sudo apt-get update && sudo apt-get install python3 python3-pip  # Debian/Ubuntu
sudo yum install python3 python3-pip  # RHEL/CentOS
brew install python3  # macOS
```

**Solution Windows:**
- Download from https://www.python.org/downloads/
- Ensure "Add to PATH" is checked during installation
- Restart terminal after installation

### Problem: "Permission denied"

**Solution:**
```bash
# Make scripts executable
chmod +x *.py *.sh

# Or run with python explicitly
python3 numpy_to_gguf.py
```

### Problem: Model generates gibberish

**Possible causes:**
1. **Weights not loaded correctly**
   - Reconvert from NumPy: `python3 numpy_to_gguf.py`
   - Check source file: `ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz`

2. **Model not trained**
   - Ensure you have the trained weights
   - Check weights are not all zeros: `python3 -c "import numpy as np; w=np.load('../ready-to-deploy-hf/jarvis_quantum_llm.npz'); print('std:', w['embedding.weight'].std())"`
   - Should show std > 0.001

3. **Incompatible quantization**
   - Try F32 (full precision): `python3 quantize_model.py` then use the F32 version

### Problem: Very slow generation

**Solution:**
1. **Use lighter quantization**
   ```bash
   python3 quantize_model.py
   # Creates Q4_0 version (faster, less memory)
   ```

2. **Adjust generation parameters**
   Edit `Modelfile`:
   ```
   PARAMETER num_ctx 256  # Reduce context window
   PARAMETER num_predict 50  # Limit output length
   ```

3. **Check system resources**
   ```bash
   # Monitor CPU/memory
   top
   # or
   htop
   ```

### Problem: Cannot find ollama directory

**Solution:**
```bash
# Try these common locations
ls -la ~/.ollama
ls -la $HOME/.ollama
echo $OLLAMA_MODELS

# On Windows
dir %USERPROFILE%\.ollama
```

**If truly cannot find:**
- Reinstall Ollama
- After install, run `ollama list` once to initialize directories

---

## Platform-Specific Notes

### Linux

- Default directory: `~/.ollama/models`
- Requires systemd for service: `systemctl status ollama`
- May need sudo for system-wide install

### macOS

- Default directory: `~/.ollama/models`
- Installed via Homebrew or direct download
- May need to allow app in Security & Privacy settings

### Windows

- Default directory: `%USERPROFILE%\.ollama\models`
- Runs as desktop application or service
- Use PowerShell (not CMD) for best compatibility
- Path separators: use `\` not `/` in Windows-specific commands

### WSL (Windows Subsystem for Linux)

- Ollama in WSL has separate directory from Windows
- Can't share models between WSL and Windows Ollama
- Treat as Linux installation

---

## Alternative Installation Methods

### Using Docker

If you have Docker:

```bash
# Pull Ollama image
docker pull ollama/ollama

# Run container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Copy files into container
docker cp jarvis-quantum.gguf ollama:/root/.ollama/models/blobs/
docker cp Modelfile ollama:/tmp/

# Create model inside container
docker exec -it ollama ollama create jarvis -f /tmp/Modelfile

# Use model
docker exec -it ollama ollama run jarvis
```

### Using a Different Model Name

If "jarvis" conflicts:

```bash
# Change model name
ollama create my-jarvis -f Modelfile

# Run with new name
ollama run my-jarvis
```

---

## Getting Help

### Still Having Issues?

1. **Run validation script:**
   ```bash
   python3 validate_setup.py
   ```
   This checks all components and reports specific problems.

2. **Check Ollama logs:**
   ```bash
   # Linux/Mac
   journalctl -u ollama -f
   
   # Or check ~/.ollama/logs/
   tail -f ~/.ollama/logs/server.log
   ```

3. **Enable debug mode:**
   ```bash
   OLLAMA_DEBUG=1 ollama serve
   ```

4. **Test with minimal example:**
   ```bash
   # Test with tiny prompt
   echo "Hi" | ollama run jarvis
   ```

### Report Issues

If nothing works, gather this info:
- OS and version: `uname -a` (Linux/Mac) or `ver` (Windows)
- Ollama version: `ollama --version`
- Python version: `python3 --version`
- NumPy version: `python3 -c "import numpy; print(numpy.__version__)"`
- File sizes:
  ```bash
  ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz
  ls -lh jarvis-quantum.gguf
  ```
- Error messages: Copy full error output

---

## Success Checklist

✅ Ollama installed and running  
✅ Python 3 and NumPy available  
✅ `jarvis-quantum.gguf` created successfully  
✅ `ollama create jarvis -f Modelfile` succeeded  
✅ `ollama list` shows `jarvis`  
✅ `ollama run jarvis` responds to prompts  

**If all checked, you're done! 🎉**

---

## Quick Reference Commands

```bash
# Installation
cd ollama-jarvis-setup
pip3 install numpy
python3 numpy_to_gguf.py
ollama create jarvis -f Modelfile

# Usage
ollama run jarvis
ollama list
ollama show jarvis

# Management
ollama rm jarvis           # Remove
ollama create jarvis -f Modelfile  # Recreate
ollama pull llama2         # Get another model

# Troubleshooting
ollama serve               # Start server
ollama ps                  # Show running models
python3 validate_setup.py  # Check installation
```

---

**Remember:** This is a from-scratch trained model. Every parameter was learned through real backpropagation. You're running genuine machine learning! 🎓✨

**Happy chatting with Jarvis! 🚀**
