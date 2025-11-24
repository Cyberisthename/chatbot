# Ben Lab Ollama Fine-Tuning Guide

## Overview

This directory contains everything you need to create a fine-tuned Ollama model that understands your Ben Lab system (PhaseDetector, Jarvis-5090X, Discovery Suite, etc.).

## What This Creates

A local LLM that can:
- **Explain** lab concepts (TRI, RSI, QPR-R, phase types, feature vectors)
- **Design** experiments (suggest parameters for PhaseDetector)
- **Interpret** results (understand what TRI=0.08 means)
- **Run** actual experiments (optional integration with Python code)

## Quick Start

### 1. Generate Training Data

```bash
cd /home/engine/project
python generate_lab_training_data.py
```

This creates `lab_training_data.jsonl` with instruction/output pairs from your docs.

**Expected output**: ~376 Q&A pairs covering:
- Phase types (Ising, SPT, trivial, pseudorandom)
- Discovery Suite experiments (TRI, clustering, RSI)
- Jarvis-5090X architecture (5 layers)
- Bit systems (X/Y/Z/A/S/T/C/P/R + G-graph)
- Phase MLP classifier and RL scientist workflow
- Practical how-to questions

### 2. Fine-Tune a Base Model

You need to fine-tune externally (laptop can't train). Options:

**Option A: Use Unsloth (Recommended)**
```bash
# On a cloud GPU (RunPod, Lambda, Vast, etc.)
pip install unsloth

# Fine-tune script (example)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/qwen2.5-1.5b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load your dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="lab_training_data.jsonl")

# Train with LoRA
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="instruction",
    max_seq_length=2048,
    # ... more config
)

trainer.train()
```

**Option B: Use Axolotl**
```yaml
# config.yaml
base_model: Qwen/Qwen2.5-1.5B
datasets:
  - path: lab_training_data.jsonl
    type: alpaca
lora_r: 16
lora_alpha: 32
# ... more config
```

```bash
accelerate launch -m axolotl.cli.train config.yaml
```

**Key Points**:
- Base model: Qwen2.5-1.5B, Llama-3.2-1B, or Phi-3-mini work well
- Use LoRA for cheap fine-tuning (~30 min on 1x A100)
- Train for 3-5 epochs
- Output: LoRA adapter or merged model weights

### 3. Convert to GGUF

After training, convert to GGUF for Ollama:

```bash
# If you have LoRA adapter
python convert_lora_to_gguf.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --lora-adapter ./outputs/checkpoint-final \
  --output ben-lab-qwen-1.5b.gguf

# Quantize to Q4_K_M (recommended)
./llama.cpp/quantize ben-lab-qwen-1.5b.gguf ben-lab-qwen-1.5b.Q4_K_M.gguf Q4_K_M
```

### 4. Create Ollama Model

```bash
# Copy GGUF to this directory
cp ben-lab-qwen-1.5b.Q4_K_M.gguf ollama/

# Edit Modelfile.example if needed
# Then create:
cd ollama
ollama create ben-lab -f Modelfile.example
```

### 5. Test the Model

**Chat only:**
```bash
ollama run ben-lab
```

Example queries:
- "What is TRI?"
- "Explain the four phase types"
- "How do I maximize TRI in an experiment?"

**With PhaseDetector integration:**
```bash
python ollama/ollama_lab_integration.py
```

Now you can:
```
You: Design an experiment to test Ising phase with high TRI
🤖 BEN-LAB: [LLM suggests parameters]
🔬 [Executes experiment with PhaseDetector]
✅ [Shows results]
🤖 BEN-LAB: [Interprets results]
```

## File Inventory

```
ollama/
├── README.md (this file)
├── Modelfile.example - Template for Ollama model creation
└── ollama_lab_integration.py - Python script to connect LLM ↔ PhaseDetector

lab_corpus/
├── ARCHITECTURE.md - Jarvis-5090X 5-layer stack
├── PHASE_DETECTOR.md - Phase experiments, QPR-R, feature vectors
├── PHASE_MLP_RL_TUTORIAL.md - Phase MLP classifier + RL scientist workflow
├── DISCOVERY_SUITE_README.md - TRI, clustering, RSI experiments
├── BIT_SYSTEM.md - X/Y/Z/A/S/T/C/P/R bits + G-graph
└── QUICK_REFERENCE.md - Cheat sheet of metrics, commands, and workflows

generate_lab_training_data.py - Converts docs → JSONL training pairs
```

## Training Data Quality

Check `lab_training_data.jsonl` before training:

**Good examples:**
```json
{"instruction": "What is TRI?", "output": "TRI (Time-Reversal Instability) measures..."}
{"instruction": "Design an experiment to maximize TRI.", "output": "To maximize TRI: 1. Use Ising phase..."}
```

**Add more if needed:**
Edit `generate_lab_training_data.py` → add custom Q&A in `generate_practical_qa()`.

## Model Recommendations

| Base Model | Size | Training Time | Quality | Ollama Support |
|------------|------|---------------|---------|----------------|
| Qwen2.5-1.5B | 1.5B | 20-30 min | Good | ✅ Excellent |
| Llama-3.2-1B | 1B | 15-20 min | Good | ✅ Excellent |
| Phi-3-mini | 3.8B | 45-60 min | Better | ✅ Good |
| Qwen2.5-3B | 3B | 40-50 min | Better | ✅ Excellent |

**For this lab**: Qwen2.5-1.5B is perfect (fast, cheap, Ollama-friendly).

## Advanced: Auto-Scientist Workflow

Once you have `ben-lab` model + integration script working:

```python
# In ollama_lab_integration.py
assistant = OllamaLabAssistant()

# Auto-design experiment
prompt = "Find which phase has highest TRI"
design = assistant.chat(prompt)

# Parse + execute
params = assistant.parse_experiment_request(design)
result = assistant.run_experiment(params)

# Iterate
next_prompt = f"Results: {result['summary']}. What should I try next?"
next_design = assistant.chat(next_prompt)
```

This creates a **closed loop**: LLM → experiment → results → LLM → new experiment.

## Troubleshooting

**Ollama model not found:**
```bash
ollama list  # Check if ben-lab exists
ollama pull qwen2.5:1.5b  # Test with base model first
```

**PhaseDetector import error:**
```bash
cd /home/engine/project
python -c "from jarvis5090x import PhaseDetector; print('OK')"
```

**Training: OOM (out of memory):**
- Use 4-bit quantization: `load_in_4bit=True`
- Reduce batch size: `per_device_train_batch_size=1`
- Use LoRA (don't full fine-tune)

**GGUF conversion issues:**
- Use llama.cpp's convert script: `convert-hf-to-gguf.py`
- Make sure base model + adapter are merged first

## Next Steps

1. ✅ Generate training data: `python generate_lab_training_data.py`
2. ⬆️ Upload `lab_training_data.jsonl` to cloud GPU
3. 🔥 Fine-tune with Unsloth/Axolotl (30 min)
4. ⬇️ Download fine-tuned model
5. 🔧 Convert to GGUF Q4_K_M
6. 📦 `ollama create ben-lab`
7. 🚀 `python ollama/ollama_lab_integration.py`

## Resources

- **Unsloth**: https://github.com/unslothai/unsloth
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl
- **Ollama**: https://ollama.ai/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

## Example Session

```
$ python ollama/ollama_lab_integration.py

============================================================
BEN-LAB Interactive Assistant
============================================================
Type 'exit' to quit
Type 'run <experiment>' to execute
Or just ask questions!

You: What is TRI?

🤖 BEN-LAB: TRI (Time-Reversal Instability) measures how sensitive a phase is to
time-reversal perturbations. It's calculated by running a phase experiment with
bias parameter (e.g., 0.7), then running a reversed experiment with flipped bias
(1.0 - bias = 0.3), and computing the L2 distance between feature vectors.

Low TRI (0.0001-0.001): Phase is stable under inversion
High TRI (0.1+): Phase is highly directional and time-fragile

You: run ising depth 12 bias 0.7

🤖 BEN-LAB: For an Ising symmetry-breaking phase with depth 12 and bias 0.7,
you should see:
- High magnetization in the bias direction
- Broken symmetry indicator ≈ 1.0
- Low entropy (strong ordering)
- If you measure TRI, expect value > 0.05 (directional sensitivity)

🔬 Executing: {'phase_type': 'ising_symmetry_breaking', 'depth': 12, 'bias': 0.7, 'system_size': 32, 'seed': 42}
✅ Experiment complete!
ID: ising_symmetry_breaking::abc123::def456
Summary: {'entropy_mean': 0.42, 'scrambling_score': 0.31, ...}

🤖 BEN-LAB: The results show low entropy (0.42) and low scrambling (0.31),
confirming this is an ordered phase with broken symmetry. The directional bias
of 0.7 is clearly reflected in the asymmetric feature profile. If you run TRI
with the reverse bias, you'll likely see a difference of 0.05-0.1 in the feature
vector norm.

You: tri ising

🔬 Measuring TRI for ising_symmetry_breaking...
TRI = 0.078423
Bias: 0.7, Depth: 12

You: exit
👋 Goodbye!
```

## Citation

If you use this in research:

```
We fine-tuned a domain-specific LLM on synthetic quantum lab documentation,
enabling automated experiment design and result interpretation for PhaseDetector
experiments. The model was trained on ~300 instruction/output pairs covering
phase types, discovery metrics (TRI, RSI), and Jarvis-5090X architecture.
```
