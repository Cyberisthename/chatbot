# Ben Lab: Fine-tuned Local LLM from Quantum Experiments

This guide shows you how to create **ben-lab**, a local Ollama model that learns from your quantum simulator experiments.

## What "Training with Qubits" Actually Means

You can't do gradient descent on real qubits with Ollama. But you can:

1. Use **Jarvis-2v** (PhaseDetector, TRI, RL scientist) to run thousands of experiments
2. Turn those into **Q&A / explanation pairs**
3. **Fine-tune an LLM** so its brain is shaped by patterns from your phase space

The data and behavior come from your "qubits & tech," the optimization itself runs on your GPU/CPU.

## Three-Step Pipeline

```
┌──────────────────────┐
│ 1. Generate Training │  <- Jarvis Lab API experiments
│    Data from Qubits  │
└──────────┬───────────┘
           │ data/lab_instructions.jsonl
           ▼
┌──────────────────────┐
│ 2. Fine-tune with    │  <- LoRA (Low-Rank Adaptation)
│    LoRA              │
└──────────┬───────────┘
           │ ben-lab-lora/
           ▼
┌──────────────────────┐
│ 3. Install Adapter   │  <- GGUF → Ollama
│    to Ollama         │
└──────────────────────┘
         ben-lab model
```

---

## 1. Generate Lab Training Data (Qubits → Text)

This script calls your **Jarvis Lab API** (jarvis_api.py) to run hundreds of experiments and converts them to instruction-style training samples.

### What it generates:

- **Phase experiments** - Single phase runs with various parameters
- **TRI tests** - Time-Reversal Instability measurements
- **Discovery/clustering** - Unsupervised k-means phase discovery
- **Replay drift** - Scaling experiments with depth factors

### Script: `generate_lab_training_data.py`

**Usage:**

```bash
# Make sure jarvis_api.py is running first:
python jarvis_api.py

# In another terminal:
python generate_lab_training_data.py
```

**Output:** `data/lab_instructions.jsonl`

Each line is a JSON object:
```json
{
  "instruction": "Explain the behavior of a ising_symmetry_breaking experiment with...",
  "input": "",
  "output": "Here is an analysis of the experiment:\n\n- Phase type: ising_symmetry_breaking\n...",
  "meta": {"kind": "phase", "params": {...}}
}
```

**Parameters you can adjust:**

```python
# In generate_lab_training_data.py:
generate_phase_samples(n_per_phase=40)    # Samples per phase type
generate_tri_samples(n=60)                # TRI experiments
generate_discovery_samples(n_runs=15)     # Clustering runs
generate_replay_drift_samples(n=40)       # Drift scaling experiments
```

---

## 2. Fine-tune a Small Model (LoRA)

Uses **LoRA** (Low-Rank Adaptation) to fine-tune a base model efficiently. LoRA trains only a small adapter layer, not the whole model.

### Script: `finetune_ben_lab.py`

**Requirements:**

```bash
pip install "transformers>=4.40" "datasets" "accelerate" "peft" "torch"
```

**Configuration:**

```python
# At top of finetune_ben_lab.py:
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Change as needed
OUTPUT_DIR = "ben-lab-lora"
```

**Alternative models to try:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`
- `Qwen/Qwen2.5-1.5B-Instruct`

**Usage:**

```bash
python finetune_ben_lab.py
```

**Training parameters:**

```python
TrainingArguments(
    num_train_epochs=1,              # Increase for more training
    per_device_train_batch_size=2,   # Adjust for GPU memory
    learning_rate=2e-4,
    gradient_accumulation_steps=4,
)
```

**Output:** `ben-lab-lora/` directory containing:
- LoRA adapter weights
- Tokenizer
- Configuration files

**Time estimate:** 10-30 minutes on GPU, longer on CPU

---

## 3. Convert LoRA → GGUF and Install to Ollama

### Step 3.1: Get llama.cpp and Convert

```bash
# Clone llama.cpp (if not already present)
git clone https://github.com/ggerganov/llama.cpp.git

# Convert LoRA adapter to GGUF format
python llama.cpp/scripts/convert_lora_to_gguf.py \
  --adapter-dir ben-lab-lora \
  --outfile ben-lab-adapter.gguf
```

### Step 3.2: Create Ollama Model

The `Modelfile` specifies:
- Base model to use
- Adapter to apply
- System prompt and parameters

**Modelfile:**

```dockerfile
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

**Build the model:**

```bash
ollama create ben-lab -f Modelfile
```

**Use the model:**

```bash
# Interactive chat
ollama run ben-lab

# Or in chat_with_lab.py, set:
DEFAULT_MODEL = "ben-lab"
```

---

## One-Shot Automation: `train_and_install.sh`

Runs all three steps automatically:

```bash
./train_and_install.sh
```

**What it does:**

1. ✅ Calls `generate_lab_training_data.py`
2. ✅ Calls `finetune_ben_lab.py`
3. ✅ Clones llama.cpp if needed
4. ✅ Converts adapter to GGUF
5. ✅ Creates Ollama model `ben-lab`

**Requirements:**
- Python with packages: `transformers`, `datasets`, `peft`, `torch`, `accelerate`
- Git (for llama.cpp)
- Ollama installed

---

## Verification & Usage

### Test the model:

```bash
ollama run ben-lab
```

**Try these prompts:**

```
> Explain what TRI measures in a quantum phase experiment

> What's the difference between Ising symmetry-breaking and SPT cluster phases?

> How do I interpret high vs low replay drift?

> Design an experiment to maximize TRI
```

### Use in Python:

```python
import requests

def query_ben_lab(prompt: str) -> str:
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            'model': 'ben-lab',
            'prompt': prompt,
            'stream': False
        })
    return response.json()['response']

answer = query_ben_lab("What are the four phase types in Ben Lab?")
print(answer)
```

### Integration with chat_with_lab.py:

```python
# In chat_with_lab.py, change:
DEFAULT_MODEL = "ben-lab"

# Now the chat uses your fine-tuned model
python chat_with_lab.py
```

---

## Customization

### More training data:

In `generate_lab_training_data.py`, increase sample counts:

```python
generate_phase_samples(n_per_phase=100)  # More phase experiments
generate_tri_samples(n=200)              # More TRI tests
```

### Longer training:

In `finetune_ben_lab.py`:

```python
TrainingArguments(
    num_train_epochs=3,  # Train for 3 epochs instead of 1
    ...
)
```

### Different base model:

In `finetune_ben_lab.py`:

```python
BASE_MODEL = "microsoft/phi-2"  # Use Phi-2 instead
```

In `Modelfile`:

```dockerfile
FROM phi  # Match the base model
```

---

## Troubleshooting

### "jarvis_api.py not responding"

Make sure it's running:
```bash
python jarvis_api.py
```

Check it's accessible:
```bash
curl http://127.0.0.1:8000/health
```

### "Out of memory during training"

Reduce batch size in `finetune_ben_lab.py`:

```python
per_device_train_batch_size=1,  # Reduce from 2
gradient_accumulation_steps=8,  # Increase to compensate
```

### "Model not found in Ollama"

Pull the base model first:
```bash
ollama pull llama3.2:1b
```

### "convert_lora_to_gguf.py not found"

Make sure llama.cpp is cloned:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

---

## What You Get

After running the full pipeline, you have:

✅ **ben-lab** - A local Ollama model fine-tuned on your quantum experiments  
✅ **Quantum-aware AI** - Understands phase types, TRI, clustering, drift  
✅ **No external API** - Runs entirely on your machine  
✅ **Fast inference** - Optimized GGUF format with LoRA adapter  
✅ **Customizable** - Retrain on new experiments anytime  

The model literally learned from your simulator's behavior patterns!

---

## Next Steps

1. **Generate more diverse data** - Add more phases, parameter sweeps
2. **Train longer** - Increase epochs for better performance
3. **Combine with RAG** - Use with document retrieval for hybrid system
4. **Export insights** - Use the model to explain experiment results automatically

---

## File Summary

| File | Purpose |
|------|---------|
| `generate_lab_training_data.py` | Generate training data from Jarvis API |
| `finetune_ben_lab.py` | Fine-tune model with LoRA |
| `Modelfile` | Ollama model definition |
| `train_and_install.sh` | One-shot automation script |
| `data/lab_instructions.jsonl` | Generated training data |
| `ben-lab-lora/` | LoRA adapter weights |
| `ben-lab-adapter.gguf` | GGUF-format adapter |

---

## Credits

- **LoRA**: Low-Rank Adaptation (Hu et al., 2021)
- **llama.cpp**: GGUF conversion and inference
- **Ollama**: Local LLM runtime
- **Jarvis-2v**: Quantum phase simulator
- **Ben Lab**: Your quantum experiment framework

Enjoy your quantum-trained AI! 🚀🔬
