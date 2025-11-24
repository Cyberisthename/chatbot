# 🪟 Windows Quick Start Guide

Complete guide for getting Jarvis up and running on Windows in minutes!

## 🚀 Super Easy Method (One-Click)

### Option 1: Batch File (Double-Click)

1. **Download/Clone this repository** to your Windows machine
2. **Double-click** `install_train_jarvis_windows.bat`
3. Follow the interactive prompts!

The script will automatically:
- ✅ Check if Python is installed
- ✅ Install all Python dependencies
- ✅ Check if Ollama is installed (guide you if not)
- ✅ Let you choose between quick or advanced training
- ✅ Train the Jarvis model
- ✅ Install it to Ollama for easy use

### Option 2: PowerShell (Better Logging)

1. **Open PowerShell** in the project directory
2. **Run** the script:
   ```powershell
   .\install_train_jarvis_windows.ps1
   ```
3. If you get an execution policy error:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\install_train_jarvis_windows.ps1
   ```

The PowerShell version has better colors and error handling!

## 📋 Prerequisites

### What You Need

1. **Windows 10 or Windows 11**
2. **Python 3.9+** - [Download here](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation!
3. **Ollama** (optional but recommended) - [Download here](https://ollama.ai/download)
4. **Git** (optional, for llama.cpp) - [Download here](https://git-scm.com/download/win)

### Checking Your Setup

Open Command Prompt or PowerShell and check:

```powershell
python --version     # Should show Python 3.9 or higher
pip --version        # Should show pip version
ollama --version     # Should show Ollama version (if installed)
```

## 🎯 Training Options

When you run the installer, you'll be asked to choose:

### 1. Quick Train (~5 minutes)

Best for: Testing, learning, or if you want something fast

- Trains a basic Jarvis model on simple conversational data
- Uses `train_jarvis.py`
- Output: `jarvis-model/` directory
- Quick to complete but less specialized

### 2. Lab Train (~30 minutes)

Best for: Advanced users, quantum experiments, research

- Generates training data from live quantum experiments
- Fine-tunes with LoRA (Low-Rank Adaptation)
- Creates a specialized model for lab work
- Output: `ben-lab-adapter.gguf` → installed as `ben-lab` in Ollama
- Requires Jarvis Lab API to be running

Steps:
1. Starts Jarvis Lab API (background)
2. Runs experiments to generate training data
3. Fine-tunes model with LoRA
4. Converts to GGUF format
5. Installs to Ollama

### 3. Skip Training

Best for: Just want to explore with default models

- Downloads a base Ollama model (e.g., `llama3.2:1b`)
- No training, ready to use immediately
- You can always train later

## 🛠️ Manual Installation (If Scripts Don't Work)

### Step 1: Install Python Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
python -m pip install fastapi uvicorn requests numpy matplotlib

# Install PyTorch (CPU version for Windows)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML tools
python -m pip install transformers datasets accelerate peft sentencepiece protobuf

# Install everything else
python -m pip install -r requirements.txt
```

### Step 2: Install Ollama

1. Download from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Open Command Prompt and verify:
   ```powershell
   ollama --version
   ```
4. Pull a base model:
   ```powershell
   ollama pull llama3.2:1b
   ```

### Step 3: Quick Train (Simple Method)

```powershell
python train_jarvis.py
```

This creates a `jarvis-model/` directory with your trained model.

### Step 4: Lab Train (Advanced Method)

#### 4a. Start Jarvis Lab API

Open a **new terminal** and run:
```powershell
python jarvis_api.py
```

Leave this running! You should see:
```
🚀 Starting Jarvis Lab API...
📡 Endpoints available at http://127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000
```

#### 4b. Generate Training Data

Open **another terminal**:
```powershell
python generate_lab_training_data.py
```

This will run experiments and create `data/lab_instructions.jsonl`.

#### 4c. Fine-tune Model

```powershell
python finetune_ben_lab.py
```

Output: `ben-lab-lora/` directory with LoRA adapter.

#### 4d. Convert to GGUF

First, clone llama.cpp (only needed once):
```powershell
git clone https://github.com/ggerganov/llama.cpp.git
```

Then convert:
```powershell
python llama.cpp\scripts\convert_lora_to_gguf.py --adapter-dir ben-lab-lora --outfile ben-lab-adapter.gguf
```

#### 4e. Install to Ollama

Create a `Modelfile`:
```
FROM llama3.2:1b

ADAPTER ./ben-lab-adapter.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM """
You are Ben's Lab AI (Jarvis-2v).
You understand the Jarvis-2v quantum phase simulator, TRI, replay drift, clustering,
and the lab API. You explain results clearly and use Ben's terminology.
"""
```

Then install:
```powershell
ollama create ben-lab -f Modelfile
```

## 🎮 Using Your Trained Model

### Chat with Ollama

```powershell
# If you did Lab Train
ollama run ben-lab

# Or with any other model
ollama run llama3.2:1b
```

### Use Lab Integration

Start Jarvis Lab API (if not running):
```powershell
python jarvis_api.py
```

In another terminal, start the chat interface:
```powershell
python chat_with_lab.py
```

You can now ask questions like:
- "Run an ising experiment with bias 0.7"
- "Compute TRI for spt_cluster"
- "Run discovery clustering"

### Web Interface

```powershell
python streamlit_app.py
```

Then open your browser to http://localhost:8501

## 🐛 Troubleshooting

### Problem: "Python is not recognized"

**Fix:**
1. Reinstall Python from [python.org](https://www.python.org/downloads/)
2. **CHECK** the box "Add Python to PATH"
3. Restart your terminal

### Problem: "Ollama is not recognized"

**Fix:**
1. Download Ollama from [ollama.ai](https://ollama.ai/download)
2. Install it
3. Restart your terminal
4. Test with: `ollama --version`

### Problem: PyTorch installation fails

**Fix:**
Use the CPU version explicitly:
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Scripts don't run (execution policy)

**Fix:**
In PowerShell (as Administrator):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: "Cannot find jarvis_api.py"

**Fix:**
Make sure you're in the project directory:
```powershell
cd path\to\jarvis-project
```

### Problem: Out of memory during training

**Fix:**
- Close other applications
- Use Quick Train instead of Lab Train
- Edit `finetune_ben_lab.py` and reduce batch size:
  ```python
  per_device_train_batch_size=1  # Change from 4 to 1
  ```

### Problem: Git not found

**Fix:**
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Install it
3. Restart your terminal

Or skip llama.cpp conversion and ask for help with pre-converted models.

## 📚 What's Next?

After successful installation:

1. **Explore the Lab**
   - Read `EXPERIMENTS_GUIDE.md`
   - Try quantum experiments
   - Generate more training data

2. **Customize Your Model**
   - Edit training data in `train_jarvis.py`
   - Add your own experiments
   - Retrain and improve

3. **Advanced Features**
   - Check out `PHASE_DETECTOR.md`
   - Learn about `ATOM_3D_DISCOVERY_SUMMARY.md`
   - Try the synthetic GPU miner

4. **Join the Community**
   - Share your results
   - Report issues on GitHub
   - Contribute improvements

## 🎉 Success Checklist

You're fully set up when:

- ✅ Python runs and shows version 3.9+
- ✅ `pip list` shows torch, transformers, fastapi
- ✅ Ollama runs and shows version
- ✅ `ollama list` shows at least one model
- ✅ `python jarvis_api.py` starts without errors
- ✅ `ollama run ben-lab` (or your model) works
- ✅ You can chat with the model

**Congratulations! You now have a fully trained Jarvis AI running locally on Windows!** 🎊

## 🔗 Related Documentation

- **Main README**: `README.md`
- **Lab Setup Guide**: `SETUP_JARVIS_LAB_OLLAMA.md`
- **Quick Start**: `QUICK_START_BEN_LAB_LORA.md`
- **Fine-tuning Guide**: `OLLAMA_FINETUNE_GUIDE.md`
- **Experiments**: `EXPERIMENTS_GUIDE.md`

## 💡 Tips for Windows Users

1. **Use Windows Terminal** instead of Command Prompt for a better experience
2. **PowerShell** has better features than batch files
3. **WSL2** (Windows Subsystem for Linux) can run Linux scripts if needed
4. **Virtual environments** help keep Python dependencies organized:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
5. **Task Manager** can help if processes get stuck (Ctrl+Shift+Esc)

---

**Need help?** Check the troubleshooting section above or create an issue on GitHub!
