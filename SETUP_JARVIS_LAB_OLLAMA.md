# 🚀 Jarvis Lab + Ollama Setup Guide

Complete setup instructions for running Jarvis Lab experiments with Ollama LLM integration.

## 📁 1. Verify File Structure

Your project should have these files in place:

```
/home/engine/project/
├── jarvis_api.py           ✅ FastAPI server for lab
├── chat_with_lab.py        ✅ Ollama chat bridge
├── jarvis5090x/            ✅ Lab engine (quantum detector)
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── phase_detector.py
│   └── ...
├── experiments/            ✅ Discovery suite & tools
│   ├── discovery_suite.py
│   ├── build_phase_dataset.py
│   ├── rl_scientist.py
│   └── ...
├── requirements.txt        ✅ Python dependencies
└── ollama/                 ✅ Ollama integration
    ├── README.md
    ├── Modelfile.example
    └── ollama_lab_integration.py
```

All files are correctly positioned! ✅

## 🔧 2. Install Dependencies

### Option A: Using Virtual Environment (Recommended)

#### On Linux/Mac:
```bash
cd /home/engine/project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn requests
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### On Windows:
```bash
cd C:\path\to\project
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install fastapi uvicorn requests
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Option B: Using Existing Requirements
```bash
pip install -r requirements.txt
```

This installs everything including:
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `requests` - HTTP client
- `torch` - PyTorch (for ML components)
- Plus all other lab dependencies

### CPU-Only PyTorch (if no GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 🧠 3. Install & Start Ollama

### Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Mac:**
```bash
brew install ollama
```

**Windows:**
Download from: https://ollama.ai/download

### Start Ollama Server

Open a **separate terminal** and run:
```bash
ollama serve
```

Leave this running in the background.

### Pull a Model

In another terminal:
```bash
ollama list                    # Check installed models
ollama pull llama3.1           # Pull LLaMA 3.1 (recommended)
```

**Alternative models:**
```bash
ollama pull llama3.2:1b        # Smaller, faster
ollama pull qwen2.5:1.5b       # Good for fine-tuning later
ollama pull mistral            # Alternative option
```

**Verify it works:**
```bash
ollama run llama3.1
>>> Hello! (then type /bye to exit)
```

## 🔬 4. Start the Jarvis Lab API

Open a **new terminal** and start the lab server:

```bash
cd /home/engine/project
source .venv/bin/activate      # If using venv
python jarvis_api.py
```

**Expected output:**
```
🚀 Starting Jarvis Lab API...
📡 Endpoints available at http://127.0.0.1:8000
📚 API docs at http://127.0.0.1:8000/docs
🔬 Quantum detector initialized with 1 devices
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Verify it's working:**

Open a browser and visit:
- http://127.0.0.1:8000 → Basic info (JSON)
- http://127.0.0.1:8000/docs → Interactive API docs (Swagger UI)
- http://127.0.0.1:8000/health → Health check

Or use curl:
```bash
curl http://127.0.0.1:8000/health
```

**Leave this terminal running!**

## 💬 5. Start the Chat Bridge

Open **another new terminal** and start the chat interface:

```bash
cd /home/engine/project
source .venv/bin/activate      # If using venv
python chat_with_lab.py
```

**Expected output:**
```
🤖 Connected to Jarvis Lab API at http://127.0.0.1:8000
🧠 Using Ollama model: llama3.1
Type 'exit' or 'quit' to stop.

Ben: 
```

You're now in the interactive chat loop!

## 🧪 6. Run Your First Experiments

### Example 1: Basic Phase Experiment
```
Ben: run an ising_symmetry_breaking experiment with bias 0.7 and explain what's going on

AI: TOOL: {"name": "run_phase", "args": {"phase_type": "ising_symmetry_breaking", "system_size": 32, "depth": 8, "seed": 42, "bias": 0.7}}

AI (after lab): The experiment completed successfully! This Ising symmetry-breaking phase 
shows [explanation based on feature vector and summary]...
```

### Example 2: Time-Reversal Instability (TRI)
```
Ben: compute TRI for spt_cluster with bias 0.6 and tell me if it's stable or fragile

AI: TOOL: {"name": "tri", "args": {"phase_type": "spt_cluster", "system_size": 32, "depth": 8, "bias": 0.6, "seed": 42}}

AI (after lab): The TRI value is 0.0234, which indicates this SPT cluster phase is 
relatively stable under time-reversal operations...
```

### Example 3: Phase Discovery (Clustering)
```
Ben: run a discovery clustering over all four phases and tell me what clusters you see

AI: TOOL: {"name": "discovery", "args": {"phases": ["ising_symmetry_breaking", "spt_cluster", "trivial_product", "pseudorandom"], "num_per_phase": 20, "k": 4, "iterations": 25}}

AI (after lab): The clustering results show 4 distinct clusters...
```

### Example 4: Replay Drift Scaling
```
Ben: do a replay drift scaling for ising_symmetry_breaking and describe how complexity grows with depth

AI: TOOL: {"name": "replay_drift", "args": {"phase_type": "ising_symmetry_breaking", "system_size": 32, "base_depth": 6, "seed": 123, "depth_factors": [1, 2, 3]}}

AI (after lab): The replay drift increases as depth grows: depth 6 → drift 0.0, 
depth 12 → drift 1.234, depth 18 → drift 2.567...
```

## 🎯 Available Tools

The LLM can call these tools by responding with `TOOL: {...}` format:

| Tool Name | API Endpoint | Purpose |
|-----------|--------------|---------|
| `run_phase` | `/run_phase_experiment` | Run single quantum phase experiment |
| `tri` | `/tri` | Time-Reversal Instability test (forward + reverse) |
| `discovery` | `/discovery` | Unsupervised phase clustering (k-means) |
| `replay_drift` | `/replay_drift` | Depth scaling analysis |

**Tool Format:**
```json
TOOL: {"name": "run_phase", "args": {"phase_type": "ising_symmetry_breaking", "bias": 0.7}}
```

## ⚙️ 7. Switching to Your Custom Model

Once you fine-tune a lab-specialized model (see `OLLAMA_FINETUNE_GUIDE.md`):

### Create the model in Ollama:
```bash
cd ollama/
ollama create ben-lab -f Modelfile.example
```

### Update the chat script:
Edit `chat_with_lab.py` line 13:
```python
DEFAULT_MODEL = "ben-lab"  # Changed from "llama3.1"
```

### Restart the chat bridge:
```bash
python chat_with_lab.py
```

Now you're using your custom model! 🎉

## 🐛 Troubleshooting

### Problem: Ollama connection refused
**Error:** `requests.exceptions.ConnectionError: ... 11434 ...`

**Fix:**
```bash
# Make sure Ollama is running
ollama serve

# Verify in another terminal
ollama list
```

### Problem: Lab API not reachable
**Error:** `Jarvis Lab API is not reachable. Start jarvis_api.py first.`

**Fix:**
```bash
# In one terminal, start the lab
python jarvis_api.py

# Wait for "Uvicorn running on http://127.0.0.1:8000"
# Then start chat in another terminal
python chat_with_lab.py
```

### Problem: Model not found
**Error:** `Ollama returned status 404: model 'llama3.1' not found`

**Fix:**
```bash
ollama pull llama3.1
```

### Problem: ModuleNotFoundError
**Error:** `ModuleNotFoundError: No module named 'jarvis5090x'`

**Fix:**
```bash
# Make sure you're in the project directory
cd /home/engine/project

# Install dependencies
pip install -r requirements.txt

# Verify imports work
python -c "from jarvis5090x import PhaseDetector; print('✅ OK')"
```

### Problem: Import torch failed
**Error:** PyTorch import errors or CUDA warnings

**Fix:**
PyTorch is optional for most features. If you don't need ML components:
```bash
# The lab will work without it (falls back to k-NN classifiers)
# But if you want it:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: LLM doesn't use TOOL format
**Issue:** AI just talks instead of calling experiments

**Fix:**
Nudge it with a direct request:
```
Ben: Use the TOOL format to actually run the experiment, then explain the result.
```

The system prompt in `chat_with_lab.py` already instructs the model, but sometimes it needs reminding.

## 📊 Test the Lab Directly

You can test the lab API without Ollama:

### Using curl:
```bash
# Basic health check
curl http://127.0.0.1:8000/health

# Run a phase experiment
curl -X POST http://127.0.0.1:8000/run_phase_experiment \
  -H "Content-Type: application/json" \
  -d '{"phase_type": "ising_symmetry_breaking", "bias": 0.7}'
```

### Using Python:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/run_phase_experiment",
    json={"phase_type": "ising_symmetry_breaking", "bias": 0.7}
)
print(response.json())
```

### Using the discovery suite directly:
```bash
cd experiments/
python discovery_suite.py
```

Expected output:
```
=== EXPERIMENT A: Time-Reversal Instability ===
ising_symmetry_breaking   TRI = 0.523456
spt_cluster              TRI = 0.123456
...

=== EXPERIMENT B: Unsupervised Phase Discovery ===
Cluster 0: {'ising_symmetry_breaking': 28, 'spt_cluster': 2}
...
```

## 🔄 Typical Workflow

### Option A: All-in-one launcher
```bash
cd /home/engine/project
./start_lab_chat.sh
```
This script verifies your setup, ensures Ollama is running, starts the lab API, and
launches the chat interface. Press `Ctrl+C` to stop both services.

### Option B: Manual terminals

#### Terminal 1: Ollama Server
```bash
ollama serve
# Leave running
```

#### Terminal 2: Jarvis Lab API
```bash
cd /home/engine/project
source .venv/bin/activate
python jarvis_api.py
# Leave running
```

#### Terminal 3: Chat Interface
```bash
cd /home/engine/project
source .venv/bin/activate
python chat_with_lab.py
# Interactive session
```

## 📚 Next Steps

1. ✅ **Collect experiment logs** - Save interesting Q&A pairs from chat sessions
2. 🔬 **Generate training data** - Run `python generate_lab_training_data.py`
3. 🧠 **Fine-tune a model** - See `OLLAMA_FINETUNE_GUIDE.md` and `ollama/README.md`
4. 🚀 **Deploy as `ben-lab`** - Your custom lab-specialized LLM!
5. 🔄 **Iterate** - The more experiments you run, the smarter your model gets

## 📖 Related Documentation

- **Lab API Details:** `JARVIS_OLLAMA_BRIDGE.md`
- **Fine-tuning Guide:** `OLLAMA_FINETUNE_GUIDE.md`
- **Ollama Integration:** `ollama/README.md`
- **Phase Detector:** `PHASE_DETECTOR.md`
- **Discovery Suite:** `experiments/DISCOVERY_SUITE_README.md`
- **Experiments Guide:** `EXPERIMENTS_GUIDE.md`

## 🎉 Success Indicators

You're fully set up when:

- ✅ Ollama responds to `ollama list`
- ✅ Lab API returns 200 at http://127.0.0.1:8000/health
- ✅ Chat interface connects and displays `Ben:` prompt
- ✅ LLM can call tools with `TOOL: {...}` format
- ✅ Lab experiments execute and return results
- ✅ LLM interprets results and provides explanations

**You now have a language-model-powered quantum phase lab!** 🚀🔬

---

## Quick Reference Card

```bash
# Start everything (3 terminals):
Terminal 1: ollama serve
Terminal 2: python jarvis_api.py
Terminal 3: python chat_with_lab.py

# Test components:
curl http://127.0.0.1:8000/health       # Lab health
ollama list                              # Ollama models
python experiments/discovery_suite.py    # Direct lab test

# Example chat commands:
Ben: run ising experiment with bias 0.8
Ben: compute TRI for spt_cluster
Ben: run discovery clustering
Ben: replay drift scaling for pseudorandom phase
```

**Ready to run experiments!** 🎯
