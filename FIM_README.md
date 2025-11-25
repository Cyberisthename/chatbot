# Fill-in-the-Middle Code Completion Model

## Quick Start

Train a code completion model and export to GGUF in one command:

```bash
./train_fim_and_export.sh
```

This produces:
- ✅ `fim-model-q4_0.gguf` (~1.4 GB) - Ready-to-use quantized model
- ✅ `Modelfile.fim` - Ollama configuration
- ✅ Training data and checkpoints

## Install & Use

```bash
# Install in Ollama
ollama create fim-code-completion -f Modelfile.fim

# Test it
ollama run fim-code-completion

# Automated tests
python test_fim_model.py
```

## What is FIM?

Fill-in-the-Middle models complete code using **both** prefix and suffix context:

```python
# PREFIX (before cursor)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    [CURSOR HERE]
    
# SUFFIX (after cursor)
        if arr[mid] == target:
            return mid
```

The model intelligently fills in: `while left <= right:\n        mid = (left + right) // 2`

## Documentation

- 📖 **[QUICK_START_FIM.md](QUICK_START_FIM.md)** - Quick reference and examples
- 📚 **[FIM_MODEL_GUIDE.md](FIM_MODEL_GUIDE.md)** - Complete guide with IDE integration
- 📝 **[FIM_IMPLEMENTATION_SUMMARY.md](FIM_IMPLEMENTATION_SUMMARY.md)** - Technical details

## Usage Example

```python
import subprocess

def complete_code(prefix, suffix):
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    result = subprocess.run(
        ['ollama', 'run', 'fim-code-completion', prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Example
prefix = "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
suffix = "\n    return fibonacci(n-1) + fibonacci(n-2)"
print(complete_code(prefix, suffix))
```

## Features

- ✅ Multi-language support (Python, JavaScript, SQL)
- ✅ Context-aware completions using prefix + suffix
- ✅ LoRA fine-tuning for efficiency
- ✅ GGUF export for production deployment
- ✅ Ollama integration
- ✅ Automated testing suite
- ✅ IDE-ready API

## Requirements

```bash
pip install transformers>=4.40 datasets accelerate peft torch
```

Optional: `ollama` for easy deployment

## Training Data

The script generates 2000+ training examples from:
- **Python** (60%): algorithms, ML, web APIs
- **JavaScript** (30%): async, React, utilities
- **SQL** (10%): complex queries

Easily extensible to add more languages or your own code.

## Model Specs

| Spec | Value |
|------|-------|
| Base Model | Llama-3.2-1B-Instruct |
| Parameters | 1.2B |
| Quantization | 4-bit (Q4_0) |
| Size | ~1.4 GB |
| Speed | 15-20 tok/s (CPU) |

## Need Help?

1. Check [QUICK_START_FIM.md](QUICK_START_FIM.md) for common use cases
2. Read [FIM_MODEL_GUIDE.md](FIM_MODEL_GUIDE.md) for detailed info
3. Run `python test_fim_model.py` for diagnostics

---

**Ready to train? Run:** `./train_fim_and_export.sh`
