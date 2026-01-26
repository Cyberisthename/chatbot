# 🔧 Troubleshooting Guide

**Quick solutions to common problems**

---

## 🚨 Quick Diagnostic

Run this first to identify issues:

```bash
cd ollama-jarvis-setup
python3 validate_setup.py
```

This checks:
- ✅ Prerequisites installed
- ✅ Files present
- ✅ Model registered
- ✅ Weights valid

If all pass, your setup is correct. If not, see specific errors below.

---

## 📋 Common Problems & Solutions

### 1. "ollama: command not found"

**Problem:** Ollama is not installed or not in PATH.

**Solution:**

**Install Ollama:**
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

**Add to PATH (if installed but not found):**
```bash
# Check where Ollama is
which ollama
ls -l /usr/local/bin/ollama

# Add to PATH temporarily
export PATH=$PATH:/usr/local/bin

# Add to PATH permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc
```

**Verify:**
```bash
ollama --version
```

---

### 2. "model 'jarvis' not found"

**Problem:** Model not created or registered in Ollama.

**Solution 1: Create the model**
```bash
cd ollama-jarvis-setup
ollama create jarvis -f Modelfile
```

**Solution 2: Check if model exists with different name**
```bash
ollama list
# Look for jarvis or similar names
```

**Solution 3: Recreate from scratch**
```bash
# Remove old (if exists)
ollama rm jarvis 2>/dev/null

# Convert model
python3 numpy_to_gguf.py

# Create fresh
ollama create jarvis -f Modelfile

# Verify
ollama list | grep jarvis
```

---

### 3. "failed to load model"

**Problem:** GGUF file corrupted, incomplete, or wrong format.

**Solution 1: Reconvert**
```bash
cd ollama-jarvis-setup

# Delete old GGUF
rm jarvis-quantum.gguf

# Reconvert
python3 numpy_to_gguf.py

# Check file size (should be ~45-50MB)
ls -lh jarvis-quantum.gguf

# Recreate model
ollama rm jarvis 2>/dev/null
ollama create jarvis -f Modelfile
```

**Solution 2: Check source weights**
```bash
# Verify NumPy weights exist
ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz

# Check weights are valid (not empty)
python3 -c "
import numpy as np
data = np.load('../ready-to-deploy-hf/jarvis_quantum_llm.npz')
print('Keys:', list(data.keys()))
print('Shape:', data['embedding.weight'].shape)
print('Std:', data['embedding.weight'].std())
print('Valid:', data['embedding.weight'].std() > 0.001)
"
```

If std is near 0, weights are not trained. You need to train the model first.

---

### 4. "cannot connect to ollama server"

**Problem:** Ollama server not running.

**Solution Linux/Mac:**
```bash
# Start server
ollama serve &

# Or with systemd
sudo systemctl start ollama
sudo systemctl enable ollama

# Verify server is running
curl http://localhost:11434/api/tags

# Check process
ps aux | grep ollama
```

**Solution Windows:**
```powershell
# Start Ollama desktop app
# Or run in PowerShell:
Start-Process ollama serve

# Verify
curl http://localhost:11434/api/tags
```

**Solution Docker:**
```bash
# Start Ollama container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Check container
docker ps | grep ollama
```

---

### 5. "ModuleNotFoundError: No module named 'numpy'"

**Problem:** NumPy not installed.

**Solution:**
```bash
# Install NumPy
pip3 install numpy

# If pip3 not found, try pip
pip install numpy

# Verify
python3 -c "import numpy; print(numpy.__version__)"

# If using conda
conda install numpy

# If permissions issue
pip3 install --user numpy
```

---

### 6. "python3: command not found"

**Problem:** Python 3 not installed.

**Solution:**

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install python3 python3-pip

# RHEL/CentOS/Fedora
sudo yum install python3 python3-pip
# or
sudo dnf install python3 python3-pip

# Verify
python3 --version
```

**macOS:**
```bash
# Using Homebrew
brew install python3

# Verify
python3 --version
```

**Windows:**
- Download from https://www.python.org/downloads/
- Run installer
- ✅ Check "Add Python to PATH"
- Restart terminal
- Verify: `python --version`

---

### 7. "FileNotFoundError: jarvis_quantum_llm.npz"

**Problem:** Model weights not found.

**Solution:**
```bash
# Check if file exists
ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz

# If not found, check location
find .. -name "jarvis_quantum_llm.npz"

# If truly missing, you need to train the model
cd ..
python3 train_full_quantum_llm_production.py
# or use the enhanced training
cd ollama-jarvis-setup
python3 enhanced_training.py
```

---

### 8. "Modelfile parse error"

**Problem:** Syntax error in Modelfile.

**Solution:**
```bash
# Check Modelfile
cat Modelfile

# Common issues:
# 1. Wrong path to GGUF
# 2. Extra quotes
# 3. Windows line endings

# Fix line endings (if needed)
dos2unix Modelfile  # Linux/Mac
# or
sed -i 's/\r$//' Modelfile

# Verify GGUF path
head -1 Modelfile
# Should be: FROM ./jarvis-quantum.gguf

# Verify GGUF exists
ls -l jarvis-quantum.gguf
```

**If path is wrong:**
```bash
# Edit Modelfile (change first line)
sed -i '1s|.*|FROM ./jarvis-quantum.gguf|' Modelfile
```

---

### 9. Model Generates Gibberish

**Problem:** Model not properly trained or weights corrupted.

**Solution 1: Verify weights**
```bash
python3 -c "
import numpy as np
w = np.load('../ready-to-deploy-hf/jarvis_quantum_llm.npz')
emb = w['embedding.weight']
print('Mean:', emb.mean())
print('Std:', emb.std())
print('Min:', emb.min())
print('Max:', emb.max())
print('NaNs:', np.isnan(emb).any())
print('Infs:', np.isinf(emb).any())

# Valid weights should have:
# - Std > 0.001
# - No NaNs
# - No Infs
# - Reasonable range (e.g., -3 to +3)
"
```

**Solution 2: Try different quantization**
```bash
# Use full precision (F32)
python3 quantize_model.py

# Create model with F32 weights
# Edit Modelfile to point to F32 file
sed -i '1s|.*|FROM ./jarvis-quantum-f32.gguf|' Modelfile

ollama rm jarvis
ollama create jarvis -f Modelfile
```

**Solution 3: Retrain model**
```bash
cd ..
python3 train_full_quantum_llm_production.py
```

---

### 10. Very Slow Generation

**Problem:** Model too large or using inefficient quantization.

**Solution 1: Use Q4_0 quantization (fastest)**
```bash
cd ollama-jarvis-setup

# Create Q4_0 version
python3 quantize_model.py

# Update Modelfile
sed -i '1s|.*|FROM ./jarvis-quantum-q4_0.gguf|' Modelfile

# Recreate model
ollama rm jarvis
ollama create jarvis -f Modelfile
```

**Solution 2: Reduce context window**
Edit `Modelfile`:
```
PARAMETER num_ctx 128     # Smaller context
PARAMETER num_predict 50  # Limit output length
```

Then recreate:
```bash
ollama rm jarvis
ollama create jarvis -f Modelfile
```

**Solution 3: Check system resources**
```bash
# Monitor CPU/RAM
top
# or
htop

# Check Ollama resource usage
ps aux | grep ollama
```

---

### 11. "Permission denied" when running scripts

**Problem:** Scripts not executable.

**Solution:**
```bash
# Make all scripts executable
chmod +x *.sh *.py

# Or run with interpreter explicitly
bash setup.sh
python3 numpy_to_gguf.py
```

---

### 12. "disk space" or "no space left"

**Problem:** Not enough disk space for GGUF file.

**Solution:**
```bash
# Check available space
df -h .

# Check file sizes
ls -lh jarvis-quantum*.gguf
ls -lh ../ready-to-deploy-hf/

# Clean up old models
ollama list
ollama rm <old-model-name>

# Remove temporary files
rm -f *.tmp *.log
```

**GGUF file should be ~45-50MB. If larger, check quantization settings.**

---

### 13. Model Exists but Won't Run

**Problem:** Model registered but fails to execute.

**Solution:**
```bash
# Check model info
ollama show jarvis

# Try removing and recreating
ollama rm jarvis
cd ollama-jarvis-setup
python3 numpy_to_gguf.py
ollama create jarvis -f Modelfile

# Test with simple prompt
echo "Hi" | ollama run jarvis

# Check logs
tail -f ~/.ollama/logs/server.log  # Linux/Mac
# or check Ollama app logs on Windows
```

---

### 14. "Import Error" when running conversion

**Problem:** Missing or incompatible dependencies.

**Solution:**
```bash
# Install/upgrade all dependencies
pip3 install --upgrade numpy requests

# Or use requirements.txt
pip3 install -r requirements.txt

# Check versions
python3 -c "
import sys
print('Python:', sys.version)
import numpy
print('NumPy:', numpy.__version__)
"

# NumPy should be >= 1.19.0
# Python should be >= 3.7
```

---

### 15. "Address already in use" (Port 11434)

**Problem:** Another Ollama instance or process using port.

**Solution:**
```bash
# Find process using port
lsof -i :11434  # Linux/Mac
netstat -ano | findstr :11434  # Windows

# Kill old Ollama process
killall ollama  # Linux/Mac
taskkill /F /IM ollama.exe  # Windows

# Restart
ollama serve
```

---

## 🔍 Advanced Diagnostics

### Full System Check

```bash
#!/bin/bash
echo "=== System Diagnostics ==="
echo ""
echo "OS Info:"
uname -a
echo ""
echo "Ollama Version:"
ollama --version || echo "NOT INSTALLED"
echo ""
echo "Python Version:"
python3 --version || echo "NOT INSTALLED"
echo ""
echo "NumPy Version:"
python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NOT INSTALLED"
echo ""
echo "Disk Space:"
df -h . | tail -1
echo ""
echo "Memory:"
free -h | grep Mem || echo "N/A"
echo ""
echo "Files:"
ls -lh jarvis-quantum.gguf 2>/dev/null || echo "GGUF not found"
ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz 2>/dev/null || echo "NPZ not found"
echo ""
echo "Ollama Models:"
ollama list
echo ""
echo "Ollama Server Status:"
curl -s http://localhost:11434/api/tags > /dev/null && echo "RUNNING" || echo "NOT RUNNING"
```

Save as `diagnose.sh` and run:
```bash
chmod +x diagnose.sh
./diagnose.sh
```

---

## 🆘 Emergency Reset

**If nothing works, start fresh:**

```bash
# 1. Stop all Ollama processes
killall ollama  # Linux/Mac
taskkill /F /IM ollama.exe  # Windows

# 2. Remove Jarvis model
ollama rm jarvis 2>/dev/null

# 3. Clean generated files
cd ollama-jarvis-setup
rm -f jarvis-quantum*.gguf

# 4. Reinstall dependencies
pip3 install --upgrade --force-reinstall numpy

# 5. Convert fresh
python3 numpy_to_gguf.py

# 6. Start Ollama
ollama serve &
sleep 2

# 7. Create model
ollama create jarvis -f Modelfile

# 8. Test
ollama run jarvis
```

---

## 📞 Getting Help

### Information to Gather

When asking for help, provide:

1. **System info:**
   ```bash
   uname -a  # Linux/Mac
   ver       # Windows
   ```

2. **Versions:**
   ```bash
   ollama --version
   python3 --version
   python3 -c "import numpy; print(numpy.__version__)"
   ```

3. **File status:**
   ```bash
   ls -lh jarvis-quantum.gguf
   ls -lh ../ready-to-deploy-hf/jarvis_quantum_llm.npz
   ```

4. **Model status:**
   ```bash
   ollama list
   ollama show jarvis
   ```

5. **Full error message:** Copy complete output including stack trace

6. **Validation output:**
   ```bash
   python3 validate_setup.py > validation.txt 2>&1
   cat validation.txt
   ```

### Check Documentation

1. **Quick Start:** `QUICK_START.md`
2. **Manual Install:** `📖_MANUAL_INSTALLATION.md`
3. **Full Guide:** `🚀_OLLAMA_JARVIS_MASTER_GUIDE.md`
4. **Technical Details:** `TECHNICAL_DETAILS.md`

---

## ✅ Verification Checklist

After fixing issues, verify:

```bash
# 1. Ollama running
ollama list
# ✅ Should list models without error

# 2. Jarvis exists
ollama list | grep jarvis
# ✅ Should show jarvis model

# 3. Files present
ls jarvis-quantum.gguf
# ✅ Should exist, ~45-50MB

# 4. Can run
echo "Test" | ollama run jarvis
# ✅ Should generate response

# 5. Weights valid
python3 -c "import numpy as np; w=np.load('../ready-to-deploy-hf/jarvis_quantum_llm.npz'); print('std:', w['embedding.weight'].std())"
# ✅ std should be > 0.001
```

**All checked? You're good to go! 🎉**

---

## 💡 Pro Tips

### Prevent Common Issues

1. **Always run from `ollama-jarvis-setup/` directory**
   ```bash
   cd ollama-jarvis-setup
   # Then run commands
   ```

2. **Check Ollama is running before operations**
   ```bash
   ollama list || ollama serve &
   ```

3. **Use absolute paths if relative paths fail**
   ```bash
   # In Modelfile
   FROM /full/path/to/jarvis-quantum.gguf
   ```

4. **Keep backup of working GGUF**
   ```bash
   cp jarvis-quantum.gguf jarvis-quantum.gguf.backup
   ```

5. **Update regularly**
   ```bash
   # Update Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Update Python packages
   pip3 install --upgrade numpy requests
   ```

---

**Remember:** This is real machine learning from scratch. Some issues are expected. Work through them systematically! 🎓

**Need more help? Check `📖_MANUAL_INSTALLATION.md` for step-by-step alternatives.**
