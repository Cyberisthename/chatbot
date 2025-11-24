# Jarvis Lab + Ollama - Complete File Index

This document provides a complete reference to all files related to the Jarvis Lab + Ollama integration.

## 🎯 Start Here

| File | Purpose | Use When |
|------|---------|----------|
| **QUICKSTART_LAB.md** | 5-minute setup guide | You want to get started ASAP |
| **SETUP_JARVIS_LAB_OLLAMA.md** | Detailed step-by-step setup | You want comprehensive instructions |
| **verify_setup.py** | Verify installation | Checking if everything is installed correctly |
| **install_lab_deps.sh** | Install Python dependencies | First-time setup or missing packages |
| **start_lab_chat.sh** | Launch everything at once | Running the lab after setup |

## 🔬 Core Lab Files

### Main Components
| File | Purpose |
|------|---------|
| **jarvis_api.py** | FastAPI server exposing lab as REST API |
| **chat_with_lab.py** | Ollama chat bridge with tool calling |

### Lab Engine
| Directory/File | Purpose |
|----------------|---------|
| **jarvis5090x/** | Quantum phase detector engine (5-layer architecture) |
| **jarvis5090x/phase_detector.py** | Main phase experiment runner |
| **jarvis5090x/orchestrator.py** | Device orchestration and scheduling |
| **jarvis5090x/phase_classifier.py** | Phase classification (k-NN, centroid) |
| **jarvis5090x/phase_mlp_classifier.py** | Neural network phase classifier (requires PyTorch) |
| **jarvis5090x/quantum_layer.py** | Quantum bit system (X/Y/Z/A/S/T/C/P/R) |

### Experiments
| Directory/File | Purpose |
|----------------|---------|
| **experiments/discovery_suite.py** | TRI, clustering, drift scaling experiments |
| **experiments/build_phase_dataset.py** | Generate phase experiment datasets |
| **experiments/rl_scientist.py** | RL agent for experiment optimization |
| **experiments/evaluate_phase_classifiers.py** | Benchmark classifiers |

## 📚 Documentation

### Setup & Configuration
| File | Purpose |
|------|---------|
| **SETUP_JARVIS_LAB_OLLAMA.md** | Complete setup guide (7 steps) |
| **QUICKSTART_LAB.md** | Minimal 5-minute quickstart |
| **JARVIS_OLLAMA_BRIDGE.md** | Architecture and API reference |
| **LAB_INDEX.md** | This file - complete file index |

### Lab Documentation
| File | Purpose |
|------|---------|
| **PHASE_DETECTOR.md** | Phase detector system architecture |
| **EXPERIMENTS_GUIDE.md** | Experiment types and usage |
| **PHASE_MLP_RL_TUTORIAL.md** | ML classifier + RL workflow |

### Fine-tuning & Training
| File | Purpose |
|------|---------|
| **OLLAMA_FINETUNE_GUIDE.md** | Complete fine-tuning guide |
| **BEN_LAB_LORA_OLLAMA.md** | Live experiment → LoRA → Ollama pipeline |
| **QUICK_START_BEN_LAB_LORA.md** | TL;DR quickstart for Ben Lab LoRA |
| **ollama/README.md** | Ollama integration details |
| **generate_lab_training_data.py** | Generate training data from live Jarvis experiments |
| **generate_lab_doc_training_data.py** | Legacy documentation-based dataset generator |
| **finetune_ben_lab.py** | Fine-tune base model with LoRA |
| **train_and_install.sh** | One-shot automation for data → LoRA → Ollama |

## 🛠️ Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **verify_setup.py** | Check installation | `python verify_setup.py` |
| **test_ben_lab_setup.py** | Check LoRA pipeline setup | `python test_ben_lab_setup.py` |
| **install_lab_deps.sh** | Install dependencies | `./install_lab_deps.sh` |
| **start_lab_chat.sh** | Launch lab + chat | `./start_lab_chat.sh` |
| **demo_ben_lab_lora.py** | Interactive LoRA pipeline demo | `python demo_ben_lab_lora.py` |
| **train_and_install.sh** | Run full data → LoRA → Ollama pipeline | `./train_and_install.sh` |

## 📦 Ollama Integration

| Directory/File | Purpose |
|----------------|---------|
| **ollama/** | Ollama integration directory |
| **ollama/README.md** | Fine-tuning guide |
| **ollama/Modelfile.example** | Template for custom models |
| **ollama/ollama_lab_integration.py** | Alternative integration script |

## 🧪 Running Experiments

### Via Chat Interface
```bash
./start_lab_chat.sh
# Then interact via natural language
```

### Via Direct API
```bash
# Terminal 1
python jarvis_api.py

# Terminal 2 - use curl or Python
curl -X POST http://127.0.0.1:8000/run_phase_experiment \
  -H "Content-Type: application/json" \
  -d '{"phase_type": "ising_symmetry_breaking", "bias": 0.7}'
```

### Via Python Scripts
```bash
# Run discovery suite directly
cd experiments/
python discovery_suite.py

# Build datasets
python build_phase_dataset.py --num-per-phase 50

# RL scientist
python rl_scientist.py --trials 100
```

## 🎓 Learning Path

1. **Start:** `QUICKSTART_LAB.md` - Get running in 5 minutes
2. **Setup:** `SETUP_JARVIS_LAB_OLLAMA.md` - Understand the full setup
3. **Architecture:** `JARVIS_OLLAMA_BRIDGE.md` - Learn how it works
4. **Lab Details:** `PHASE_DETECTOR.md` - Understand the physics
5. **Experiments:** `EXPERIMENTS_GUIDE.md` - Run different experiments
6. **Fine-tuning (docs):** `OLLAMA_FINETUNE_GUIDE.md` - Train on documentation
7. **Fine-tuning (live):** `BEN_LAB_LORA_OLLAMA.md` - Train on live experiments with LoRA

## 🔍 Quick Reference

### Lab API Endpoints
- `GET /` - Basic info
- `GET /health` - Health check
- `POST /run_phase_experiment` - Run single experiment
- `POST /tri` - Time-Reversal Instability test
- `POST /discovery` - Unsupervised clustering
- `POST /replay_drift` - Depth scaling analysis

### Tool Names (for LLM)
- `run_phase` → Run phase experiment
- `tri` → Time-Reversal Instability
- `discovery` → Phase clustering
- `replay_drift` → Drift scaling

### Phase Types
- `ising_symmetry_breaking` - Ordered phase with symmetry breaking
- `spt_cluster` - Symmetry-protected topological phase
- `trivial_product` - Simple product state
- `pseudorandom` - Chaotic/scrambled phase

## 📂 Directory Structure

```
/home/engine/project/
├── README.md                      # Main project README
├── QUICKSTART_LAB.md             # 5-minute quickstart
├── SETUP_JARVIS_LAB_OLLAMA.md    # Detailed setup guide
├── JARVIS_OLLAMA_BRIDGE.md       # Architecture docs
├── LAB_INDEX.md                  # This file
├── PHASE_DETECTOR.md             # Lab architecture
├── EXPERIMENTS_GUIDE.md          # Experiment reference
├── OLLAMA_FINETUNE_GUIDE.md      # Fine-tuning guide
│
├── jarvis_api.py                 # Lab API server
├── chat_with_lab.py              # Chat bridge
├── verify_setup.py               # Setup verification
├── install_lab_deps.sh           # Dependency installer
├── start_lab_chat.sh             # All-in-one launcher
├── generate_lab_training_data.py       # Training data from Jarvis API
├── generate_lab_doc_training_data.py   # Legacy doc-based dataset generator
├── finetune_ben_lab.py                 # LoRA fine-tuning script
├── train_and_install.sh                # One-shot data → LoRA → Ollama pipeline
├── BEN_LAB_LORA_OLLAMA.md              # Full live fine-tuning guide
├── QUICK_START_BEN_LAB_LORA.md         # Quickstart for LoRA pipeline
├── demo_ben_lab_lora.py                # Interactive demo of pipeline
├── test_ben_lab_setup.py               # Setup verification for LoRA pipeline
│
├── jarvis5090x/                        # Lab engine
│   ├── __init__.py
│   ├── phase_detector.py
│   ├── orchestrator.py
│   ├── phase_classifier.py
│   ├── phase_mlp_classifier.py
│   └── ...
│
├── experiments/                  # Experiments suite
│   ├── discovery_suite.py
│   ├── build_phase_dataset.py
│   ├── rl_scientist.py
│   └── ...
│
└── ollama/                       # Ollama integration
    ├── README.md
    ├── Modelfile.example
    └── ollama_lab_integration.py
```

## 🆘 Troubleshooting Map

| Problem | Solution File |
|---------|---------------|
| Installation issues | `SETUP_JARVIS_LAB_OLLAMA.md` → Troubleshooting section |
| Import errors | `verify_setup.py` → check what's missing |
| Ollama connection | `JARVIS_OLLAMA_BRIDGE.md` → Troubleshooting section |
| Lab API errors | Check `logs/jarvis_api.log` |
| Chat format issues | `SETUP_JARVIS_LAB_OLLAMA.md` → Problem: LLM doesn't use TOOL format |

## 🎯 Common Workflows

### 1. First-Time Setup
```bash
./install_lab_deps.sh      # Install dependencies
python verify_setup.py     # Verify installation
ollama pull llama3.1       # Get LLM model
./start_lab_chat.sh        # Launch everything
```

### 2. Daily Use
```bash
# Terminal 1 (if not already running)
ollama serve

# Terminal 2
./start_lab_chat.sh
```

### 3. Direct Experiments
```bash
python experiments/discovery_suite.py
python experiments/build_phase_dataset.py
python experiments/rl_scientist.py
```

### 4. Fine-tuning Pipeline (Documentation-based)
```bash
python generate_lab_doc_training_data.py
# Upload to cloud GPU
# Fine-tune (see OLLAMA_FINETUNE_GUIDE.md)
# Download GGUF
ollama create ben-lab -f ollama/Modelfile.example
# Edit chat_with_lab.py to use ben-lab
./start_lab_chat.sh
```

### 5. Fine-tuning Pipeline (Live Experiments with LoRA)
```bash
# One-shot automation
./train_and_install.sh

# Or step-by-step
python jarvis_api.py              # Terminal 1
python generate_lab_training_data.py  # Terminal 2
python finetune_ben_lab.py
# Convert and install (see BEN_LAB_LORA_OLLAMA.md)
```

## 📊 File Dependencies

```
verify_setup.py
  ├── Checks: jarvis5090x/
  ├── Checks: experiments/
  ├── Checks: jarvis_api.py
  └── Checks: chat_with_lab.py

jarvis_api.py
  ├── Imports: jarvis5090x.*
  └── Imports: experiments.discovery_suite

chat_with_lab.py
  ├── Calls: jarvis_api.py (HTTP)
  └── Calls: Ollama (HTTP)

start_lab_chat.sh
  ├── Runs: verify_setup.py
  ├── Starts: jarvis_api.py
  └── Starts: chat_with_lab.py
```

## 🔗 External Links

- **Ollama:** https://ollama.ai/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Uvicorn:** https://www.uvicorn.org/
- **PyTorch:** https://pytorch.org/

---

**Last Updated:** 2024 (Jarvis Lab + Ollama Integration)
