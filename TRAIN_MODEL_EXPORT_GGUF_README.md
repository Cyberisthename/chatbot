# Train Model and Export to GGUF

This guide explains how to train a model on quantum lab data and export it to GGUF format for use with llama.cpp and Ollama.

## Quick Start

The simplest way to train a model and export to GGUF:

```bash
python train_and_export_gguf.py
```

This will:
1. ✅ Train a model on the existing lab training data
2. ✅ Convert to GGUF format
3. ✅ Create an Ollama Modelfile
4. ✅ (Optional) Quantize for smaller file size

## What You Get

After running the script, you'll have:

- `jarvis-lab-model/` - Trained model in HuggingFace format
- `jarvis-lab-model.gguf` - Model in GGUF format (ready for llama.cpp/Ollama)
- `jarvis-lab-model-q4.gguf` - Quantized version (smaller, faster)
- `Modelfile.jarvis-lab` - Configuration for Ollama

## Prerequisites

Install required dependencies:

```bash
pip install torch transformers datasets
```

## Usage

### Option 1: Complete Pipeline (Recommended)

Train and export everything in one go:

```bash
python train_and_export_gguf.py
```

The script will:
- Check dependencies
- Train on `lab_training_data.jsonl`
- Convert to GGUF format
- Optionally quantize
- Create Ollama Modelfile

### Option 2: Convert Existing Model

If you already have a trained model, just convert it:

```bash
python convert_to_gguf.py --model-dir my-model --output my-model.gguf
```

With quantization:

```bash
python convert_to_gguf.py --model-dir my-model --quantize --create-modelfile
```

### Option 3: Manual Steps

1. **Train a model:**
   ```bash
   python finetune_ben_lab.py
   ```

2. **Convert to GGUF:**
   ```bash
   python convert_to_gguf.py --model-dir ben-lab-lora
   ```

3. **Create Ollama model:**
   ```bash
   ollama create jarvis-lab -f Modelfile.jarvis-lab
   ```

## Training Data

The script uses `lab_training_data.jsonl` which contains ~300+ training samples about:

- Quantum phase experiments (Ising, SPT, trivial, pseudorandom)
- Time-Reversal Instability (TRI) measurements
- Unsupervised clustering and phase discovery
- Replay drift analysis
- Phase detector API usage

### Generate New Training Data

To create fresh training data from the API:

```bash
# Start the Jarvis API
python jarvis_api.py

# In another terminal, generate training data
python generate_lab_training_data.py
```

This creates `data/lab_instructions.jsonl` with live experimental results.

## GGUF Format

GGUF (GPT-Generated Unified Format) is a binary format for storing language models, optimized for:

- ✅ Fast loading and inference with llama.cpp
- ✅ Memory-mapped file access
- ✅ Quantization support (Q4, Q5, Q8 etc.)
- ✅ Compatible with Ollama, llama.cpp, and many other tools

## Using the Exported Model

### With Ollama

```bash
# Create the model
ollama create jarvis-lab -f Modelfile.jarvis-lab

# Chat with it
ollama run jarvis-lab

# Use in your code
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis-lab",
  "prompt": "What is TRI in quantum phase detection?"
}'
```

### With llama.cpp

```bash
# Clone llama.cpp if you haven't
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run the model
./llama-cli -m ../jarvis-lab-model.gguf -p "Explain SPT phases"
```

### With Python (llama-cpp-python)

```python
from llama_cpp import Llama

model = Llama(model_path="./jarvis-lab-model.gguf")

response = model("What is time-reversal instability?", max_tokens=256)
print(response['choices'][0]['text'])
```

## Quantization

Quantization reduces model size with minimal quality loss:

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| F16    | 100% | Best    | Full precision |
| Q8_0   | ~50% | Excellent | High quality, smaller |
| Q5_1   | ~35% | Very good | Balanced |
| Q4_0   | ~25% | Good | Fast inference |
| Q3_K_S | ~15% | Fair | Smallest size |

To quantize manually:

```bash
cd llama.cpp
make llama-quantize

./llama-quantize ../jarvis-lab-model.gguf ../jarvis-lab-q4.gguf Q4_0
```

## Customization

### Change Base Model

Edit `train_and_export_gguf.py`:

```python
train_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # or any HF model
    # ...
)
```

### Adjust Training Parameters

```python
train_model(
    epochs=5,           # More training
    batch_size=4,       # Larger batches (needs more memory)
    max_length=1024,    # Longer contexts
)
```

### Use Different Training Data

```python
train_model(
    data_path="my_custom_data.jsonl",
    # ...
)
```

## Troubleshooting

### "PyTorch not found"

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Or for GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "Training data not found"

Make sure `lab_training_data.jsonl` exists. If not, generate it:

```bash
python generate_lab_doc_training_data.py
```

### "llama.cpp conversion failed"

The script automatically clones llama.cpp. If it fails:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
```

### Out of Memory

Reduce batch size or use a smaller model:

```python
train_model(
    model_name="distilgpt2",  # Smaller model
    batch_size=1,              # Minimum batch size
)
```

## Architecture

```
Training Data (JSONL)
        ↓
    Tokenization
        ↓
  HuggingFace Trainer
        ↓
Trained Model (HF format)
        ↓
llama.cpp convert_hf_to_gguf.py
        ↓
    GGUF File
        ↓
  llama-quantize (optional)
        ↓
  Quantized GGUF
        ↓
  Ollama / llama.cpp
```

## Advanced Usage

### Convert LoRA Adapter to GGUF

If you trained with LoRA:

```bash
# First, merge LoRA adapter with base model
python merge_lora_adapter.py --base-model meta-llama/Llama-3.2-1B --adapter ben-lab-lora --output merged-model

# Then convert
python convert_to_gguf.py --model-dir merged-model --quantize
```

### Batch Conversion

Convert multiple models:

```bash
for model in model-*; do
    python convert_to_gguf.py --model-dir "$model" --quantize
done
```

### Custom Modelfile

Edit `Modelfile.jarvis-lab` to customize:

```
FROM ./jarvis-lab-model.gguf

PARAMETER temperature 0.3        # More deterministic
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """
Your custom system prompt here
"""

MESSAGE user "What is quantum phase detection?"
MESSAGE assistant "I'm an expert in quantum phases..."
```

## Related Files

- `train_and_export_gguf.py` - Complete training and export pipeline
- `convert_to_gguf.py` - Standalone GGUF conversion
- `finetune_ben_lab.py` - LoRA fine-tuning script
- `generate_lab_training_data.py` - Generate training data from API
- `train_and_install.sh` - Bash script for complete workflow

## See Also

- [BEN_LAB_LORA_OLLAMA.md](BEN_LAB_LORA_OLLAMA.md) - LoRA fine-tuning guide
- [PHASE_MLP_RL_SCIENTIST_README.md](PHASE_MLP_RL_SCIENTIST_README.md) - Phase classifier ML
- [llama.cpp documentation](https://github.com/ggerganov/llama.cpp)
- [Ollama documentation](https://github.com/ollama/ollama)
