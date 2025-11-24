# Quick Start: Ben Lab Fine-tuned LLM

Train a local Ollama model on your quantum experiments in 3 steps.

## Prerequisites

```bash
# Install Python packages
pip install "transformers>=4.40" "datasets" "accelerate" "peft" "torch" "requests"

# Install Ollama
# Visit: https://ollama.ai/download
```

## Three Commands to Fine-tune

### 1. Generate Training Data

```bash
# Start Jarvis API (terminal 1)
python jarvis_api.py

# Generate training data (terminal 2)
python generate_lab_training_data.py
```

Output: `data/lab_instructions.jsonl`

### 2. Fine-tune Model

```bash
python finetune_ben_lab.py
```

Output: `ben-lab-lora/` adapter

### 3. Install to Ollama

```bash
# Get llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git

# Convert adapter to GGUF
python llama.cpp/scripts/convert_lora_to_gguf.py \
  --adapter-dir ben-lab-lora \
  --outfile ben-lab-adapter.gguf

# Create Ollama model
ollama create ben-lab -f Modelfile
```

## Or Use One-Shot Script

```bash
./train_and_install.sh
```

This runs all three steps automatically.

## Test Your Model

```bash
ollama run ben-lab
```

**Try these prompts:**
- "What is TRI and how does it measure phase properties?"
- "Explain the four phase types in Ben Lab"
- "How do I design an experiment to maximize TRI?"

## Integration

Use in your Python code:

```python
import ollama

response = ollama.generate(
    model='ben-lab',
    prompt='Explain SPT cluster phases'
)
print(response['response'])
```

Or set as default in `chat_with_lab.py`:

```python
DEFAULT_MODEL = "ben-lab"
```

## Files Created

- `generate_lab_training_data.py` - Generate data from experiments
- `finetune_ben_lab.py` - Fine-tune with LoRA
- `Modelfile` - Ollama model definition
- `train_and_install.sh` - Automation script
- `BEN_LAB_LORA_OLLAMA.md` - Full documentation

## What You Get

✅ Local LLM trained on your quantum experiments  
✅ Understands phase types, TRI, clustering, drift  
✅ No external API calls  
✅ Fast inference with GGUF format  
✅ Retrain anytime with new experiments  

See `BEN_LAB_LORA_OLLAMA.md` for full details and troubleshooting.
