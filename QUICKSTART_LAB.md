# 🚀 Jarvis Lab + Ollama - 5-Minute Quickstart

The absolute fastest way to get your quantum phase lab running with LLM integration.

## Prerequisites

- Python 3.9+
- Internet connection (to download Ollama)

## Step 1: Clone or navigate to this project

```bash
cd /home/engine/project
# or wherever you have Jarvis Lab
```

## Step 2: Install Python dependencies

```bash
./install_lab_deps.sh
```

This installs: `fastapi`, `uvicorn`, `requests`, `numpy`, and optionally `torch`.

## Step 3: Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Mac:**
```bash
brew install ollama
```

**Windows:**  
Download from https://ollama.ai/download

## Step 4: Start Ollama and pull a model

In a terminal:
```bash
ollama serve
```

In another terminal:
```bash
ollama pull llama3.1
```

## Step 5: Launch everything

```bash
./start_lab_chat.sh
```

That's it! The script will:
- ✅ Verify Ollama is running
- ✅ Check your setup
- ✅ Start the Jarvis Lab API
- ✅ Launch the chat interface

## Try it out

```
Ben: run an ising_symmetry_breaking experiment with bias 0.7 and explain what's going on

AI: TOOL: {"name": "run_phase", "args": {"phase_type": "ising_symmetry_breaking", ...}}

AI (after lab): The experiment completed! The Ising symmetry-breaking phase shows...
```

## What can you ask?

- "Run an Ising experiment with bias 0.8"
- "Compute TRI for spt_cluster with bias 0.6"
- "Run discovery clustering over all four phases"
- "Do replay drift scaling for pseudorandom phase"

## File structure overview

```
/home/engine/project/
├── jarvis_api.py              # Lab API server (FastAPI)
├── chat_with_lab.py           # Chat bridge (Ollama ↔ Lab)
├── jarvis5090x/               # Quantum phase detector engine
├── experiments/               # Discovery suite, RL, datasets
├── install_lab_deps.sh        # Dependency installer
├── start_lab_chat.sh          # All-in-one launcher
├── verify_setup.py            # Setup verification
├── SETUP_JARVIS_LAB_OLLAMA.md # Detailed setup guide
└── JARVIS_OLLAMA_BRIDGE.md    # Architecture docs
```

## Troubleshooting

**Problem: "Ollama is not running"**
```bash
# Start Ollama in a separate terminal
ollama serve
```

**Problem: "No module named 'fastapi'"**
```bash
# Install dependencies
./install_lab_deps.sh
# or
pip install fastapi uvicorn requests numpy
```

**Problem: "Setup verification failed"**
```bash
# Check what's missing
python verify_setup.py
```

**Problem: Chat doesn't use TOOL format**
```
Ben: Use the TOOL format to actually run the experiment, then explain the result.
```

## What's next?

1. ✅ Run experiments via chat
2. 🔬 Collect interesting Q&A pairs
3. 🧠 Generate training data: `python generate_lab_training_data.py`
4. 🚀 Fine-tune your own model (see `OLLAMA_FINETUNE_GUIDE.md`)

## Full documentation

- **Setup guide:** `SETUP_JARVIS_LAB_OLLAMA.md`
- **Architecture:** `JARVIS_OLLAMA_BRIDGE.md`
- **Fine-tuning:** `OLLAMA_FINETUNE_GUIDE.md`
- **Lab details:** `PHASE_DETECTOR.md`, `EXPERIMENTS_GUIDE.md`

---

**You're now running a language-model-powered quantum phase lab!** 🚀🔬
