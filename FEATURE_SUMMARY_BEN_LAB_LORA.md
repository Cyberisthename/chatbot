# Feature Summary: Ben Lab LoRA Fine-tuning Pipeline

## Overview

This feature enables fine-tuning a local LLM on quantum experiments from the Jarvis Lab API, creating a specialized "ben-lab" model in Ollama that understands quantum phases, TRI measurements, and lab terminology.

## What's New

### 🚀 Core Pipeline Scripts

1. **`generate_lab_training_data.py`** (9KB, executable)
   - Generates JSONL training data from live Jarvis Lab API experiments
   - Runs 300+ experiments across 4 categories:
     - Phase experiments (40 per phase × 4 phases = 160)
     - TRI tests (60)
     - Discovery/clustering (15)
     - Replay drift scaling (40)
   - Output: `data/lab_instructions.jsonl`

2. **`finetune_ben_lab.py`** (6.5KB, executable)
   - Fine-tunes base model (default: Llama-3.2-1B) using LoRA
   - Configurable: model, batch size, epochs, learning rate
   - Saves adapter to `ben-lab-lora/` directory
   - Memory-efficient: trains only adapter layers (~1-5M params)

3. **`train_and_install.sh`** (1.5KB, executable)
   - One-shot automation script
   - Runs all 4 steps:
     1. Generate training data
     2. Fine-tune with LoRA
     3. Convert adapter to GGUF via llama.cpp
     4. Create Ollama model
   - Handles missing dependencies gracefully

4. **`Modelfile`** (805B)
   - Ollama model definition
   - Specifies base model, adapter, parameters, system prompt
   - Ready to use with `ollama create ben-lab -f Modelfile`

### 📚 Documentation

5. **`BEN_LAB_LORA_OLLAMA.md`** (8.9KB)
   - Complete guide to the pipeline
   - Explains "training with qubits" concept
   - Detailed steps for each phase
   - Customization options
   - Troubleshooting guide
   - Sample prompts for testing

6. **`QUICK_START_BEN_LAB_LORA.md`** (2.1KB)
   - TL;DR version
   - Three commands to fine-tune
   - Quick integration examples
   - Essential information only

### 🛠️ Utilities

7. **`demo_ben_lab_lora.py`** (7.7KB, executable)
   - Interactive demo of the pipeline
   - Shows each step with explanations
   - Demonstrates data generation from API
   - Explains LoRA concept
   - Sample prompts and usage examples

8. **`test_ben_lab_setup.py`** (5.7KB, executable)
   - Comprehensive setup verification
   - Checks:
     - Python packages (transformers, datasets, peft, torch, etc.)
     - Jarvis API accessibility
     - Script existence and permissions
     - Ollama installation
   - Provides clear pass/fail summary

### 🔄 Legacy Rename

9. **`generate_lab_doc_training_data.py`** (renamed from `generate_lab_training_data.py`)
   - Original documentation-based training data generator
   - Preserved for compatibility

## Updated Files

### `README.md`
- Added new section: "Fine-tune Ben Lab LLM from Live Experiments"
- Links to both full guide and quick start

### `LAB_INDEX.md`
- Added entries for all new files in appropriate sections:
  - Fine-tuning & Training
  - Utility Scripts
  - Directory Structure
- Added new workflow: "Fine-tuning Pipeline (Live Experiments with LoRA)"
- Updated learning path with step 7

### `generate_lab_training_data.py`
- Replaced with live API experiment version
- Was: documentation-based generator
- Now: calls Jarvis API to run real experiments

## Technical Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User Experience                       │
├──────────────────────────────────────────────────────────┤
│  ./train_and_install.sh   → One command to fine-tune     │
│  OR                                                       │
│  Individual steps         → Full control                  │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│                  Data Generation Layer                    │
├──────────────────────────────────────────────────────────┤
│  generate_lab_training_data.py                            │
│    ├─→ Calls jarvis_api.py endpoints                     │
│    ├─→ Runs 300+ quantum experiments                     │
│    └─→ Converts to instruction/output pairs              │
│                                                           │
│  Output: data/lab_instructions.jsonl                      │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│                   Fine-tuning Layer                       │
├──────────────────────────────────────────────────────────┤
│  finetune_ben_lab.py                                      │
│    ├─→ Loads base model (e.g., Llama-3.2-1B)            │
│    ├─→ Applies LoRA adapter                              │
│    ├─→ Trains on quantum experiment data                 │
│    └─→ Saves adapter weights                             │
│                                                           │
│  Output: ben-lab-lora/ directory                          │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│                  Conversion Layer                         │
├──────────────────────────────────────────────────────────┤
│  llama.cpp/convert_lora_to_gguf.py                       │
│    ├─→ Converts HuggingFace adapter                      │
│    └─→ Outputs GGUF format                               │
│                                                           │
│  Output: ben-lab-adapter.gguf                             │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│                  Deployment Layer                         │
├──────────────────────────────────────────────────────────┤
│  Modelfile + ollama create                                │
│    ├─→ Specifies base model                              │
│    ├─→ Attaches adapter                                  │
│    └─→ Configures system prompt                          │
│                                                           │
│  Output: "ben-lab" model in Ollama                        │
└──────────────────────────────────────────────────────────┘
```

## Dependencies

### Required Python Packages
- `requests` - API communication
- `transformers>=4.40` - Model loading and training
- `datasets` - Dataset management
- `peft` - LoRA implementation
- `torch` - Deep learning backend
- `accelerate` - Distributed training support

### External Tools
- `ollama` - Local LLM runtime (optional for final deployment)
- `git` - For cloning llama.cpp

### Install Command
```bash
pip install requests transformers>=4.40 datasets peft torch accelerate
```

## Usage Patterns

### Pattern 1: Quick Start (One Command)
```bash
./train_and_install.sh
```

### Pattern 2: Step-by-Step (Full Control)
```bash
# Terminal 1: Start API
python jarvis_api.py

# Terminal 2: Generate and train
python generate_lab_training_data.py
python finetune_ben_lab.py

# Convert and install
git clone https://github.com/ggerganov/llama.cpp.git
python llama.cpp/scripts/convert_lora_to_gguf.py \
  --adapter-dir ben-lab-lora \
  --outfile ben-lab-adapter.gguf
ollama create ben-lab -f Modelfile
```

### Pattern 3: Testing Before Training
```bash
python test_ben_lab_setup.py  # Check dependencies
python demo_ben_lab_lora.py   # Interactive walkthrough
```

## What the User Gets

After running the pipeline:

✅ **ben-lab** - Local Ollama model trained on quantum experiments  
✅ **Domain expertise** - Understands phase types, TRI, clustering, drift  
✅ **No cloud required** - Entire pipeline runs locally  
✅ **Efficient training** - LoRA trains only 1-5M parameters  
✅ **Fast inference** - GGUF format optimized for speed  
✅ **Retrainable** - Easy to update with new experiments  

## File Organization

```
project/
├── generate_lab_training_data.py        # NEW: Live experiment data gen
├── finetune_ben_lab.py                  # NEW: LoRA fine-tuning
├── train_and_install.sh                 # NEW: One-shot automation
├── Modelfile                            # NEW: Ollama model definition
├── demo_ben_lab_lora.py                 # NEW: Interactive demo
├── test_ben_lab_setup.py                # NEW: Setup verification
├── BEN_LAB_LORA_OLLAMA.md              # NEW: Full guide
├── QUICK_START_BEN_LAB_LORA.md         # NEW: Quick reference
├── generate_lab_doc_training_data.py    # RENAMED: Legacy doc generator
├── README.md                            # UPDATED: Added section
├── LAB_INDEX.md                         # UPDATED: Added entries
└── ... (existing files)
```

## Integration Points

### With Existing Lab
- Calls `jarvis_api.py` endpoints
- Uses `PhaseDetector` for experiments
- Leverages `discovery_suite.py` functions

### With Ollama
- Creates model via `ollama create`
- Compatible with `chat_with_lab.py` by setting `DEFAULT_MODEL = "ben-lab"`
- Works with existing Ollama tools and API

### With Training Ecosystem
- Compatible with HuggingFace Transformers
- Uses PEFT for efficient LoRA training
- Converts to llama.cpp GGUF format

## Metrics

- **Files added:** 8 new files
- **Files modified:** 3 files  
- **Total code:** ~35KB of new Python/Bash
- **Documentation:** ~11KB of new markdown
- **Training samples:** 300+ instruction/output pairs
- **Trainable parameters:** ~1-5M (LoRA adapter only)
- **Training time:** 10-30 minutes on GPU
- **Model size:** ~4-8GB (base) + ~10-50MB (adapter)

## Testing Checklist

- [x] Scripts are executable
- [x] Documentation is comprehensive
- [x] Integration with existing code verified
- [x] File permissions correct
- [x] Git status shows clean additions
- [x] README updated
- [x] LAB_INDEX updated
- [x] Legacy files preserved

## Future Enhancements

Potential improvements:
- Add support for more base models (Qwen, Phi, etc.)
- Implement automatic hyperparameter tuning
- Add evaluation metrics for fine-tuned model
- Create web UI for training pipeline
- Add multi-GPU support
- Implement model merging for multiple adapters

## Credits

- **LoRA**: Low-Rank Adaptation (Hu et al., 2021)
- **llama.cpp**: GGUF conversion and inference
- **Ollama**: Local LLM runtime
- **HuggingFace**: Transformers and PEFT libraries
- **Jarvis-2v**: Quantum phase simulator

---

**Feature Status:** ✅ Complete and ready for use  
**Branch:** `feat-ben-lab-lora-ollama`  
**Last Updated:** November 24, 2024
