# Fill-in-the-Middle (FIM) Code Completion Model

## Overview

This guide walks you through training and deploying a **Fill-in-the-Middle (FIM)** model for intelligent code completion. FIM models are specialized language models that can generate code completions given both prefix (code before cursor) and suffix (code after cursor) context, making them ideal for IDE integrations and smart code editors.

## What is FIM?

Traditional language models can only complete text given a prefix. FIM models are trained to utilize both:
- **Prefix**: Code that appears before the cursor position
- **Suffix**: Code that appears after the cursor position
- **Middle**: The completion that should be inserted

This allows for much smarter, context-aware completions in real-world coding scenarios.

### Example

```python
# Prefix (before cursor)
def fibonacci(n):
    if n <= 1:
        return n
    
    [CURSOR HERE]
    
# Suffix (after cursor)
    return fibonacci(n-1) + fibonacci(n-2)
```

A FIM model would intelligently complete this as:
```python
    # Base cases handled above, now compute recursively
```

## Quick Start

### 1. Train the FIM Model

```bash
python train_fim_model.py
```

This script will:
1. ✅ Generate 2000+ FIM training examples from diverse code samples
2. ✅ Fine-tune Llama-3.2-1B with LoRA for efficient training
3. ✅ Merge LoRA adapter with base model
4. ✅ Convert to GGUF format (4-bit quantization)
5. ✅ Create Ollama Modelfile for easy deployment

**Expected output files:**
- `data/fim_training.jsonl` - Training data
- `fim-model-lora/` - LoRA adapter weights
- `fim-model-merged/` - Full merged model
- `fim-model-q4_0.gguf` - Quantized GGUF model (~1.4 GB)
- `Modelfile.fim` - Ollama configuration

### 2. Install in Ollama

```bash
ollama create fim-code-completion -f Modelfile.fim
```

### 3. Test the Model

```bash
ollama run fim-code-completion
```

## FIM Format

The model uses special tokens to understand context:

```
<|fim_prefix|>code before cursor<|fim_suffix|>code after cursor<|fim_middle|>
```

### Example Usage

**Input:**
```
<|fim_prefix|>def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    <|fim_suffix|>
        if arr[mid] == target:
            return mid
<|fim_middle|>
```

**Model Output:**
```python
    while left <= right:
        mid = (left + right) // 2
```

## Training Data

The training pipeline generates FIM examples from:

### Code Samples by Language

1. **Python** (60%)
   - Algorithm implementations (binary search, quicksort, etc.)
   - Data structures (trees, linked lists, etc.)
   - Machine learning code (PyTorch, NumPy)
   - Web scraping and APIs

2. **JavaScript** (30%)
   - Async/await patterns
   - React components
   - Node.js utilities
   - Design patterns (debounce, retry, caching)

3. **SQL** (10%)
   - Complex queries with JOINs
   - Aggregations and window functions
   - Performance optimization patterns

### FIM Example Generation

Each code sample is split at random positions to create:
- **Prefix**: Lines 0 to split_point_1
- **Middle**: Lines split_point_1 to split_point_2
- **Suffix**: Lines split_point_2 to end

This creates realistic IDE-like completion scenarios.

## Model Architecture

### Base Model
- **Meta-Llama-3.2-1B-Instruct** (1.2B parameters)
- Pre-trained on diverse text and code
- Instruction-tuned for following prompts

### Fine-tuning Approach
- **LoRA** (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: attention + FFN projections
  - Dropout: 5%

### Training Hyperparameters
- Epochs: 3
- Batch size: 4 per device
- Gradient accumulation: 2 steps
- Learning rate: 2e-4
- Warmup: 100 steps
- Mixed precision: FP16

### Quantization
- **4-bit quantization (Q4_0)**
- ~75% size reduction
- Minimal quality loss
- Fast inference on CPU

## Advanced Usage

### Custom Training Data

Add your own code samples to improve completions:

```python
# Add to train_fim_model.py

custom_samples = [
    {
        "code": """
your custom code here
""",
        "language": "python"
    }
]

# Add to code_samples list in generate_code_samples()
```

### Adjust Training Parameters

Edit `train_fim_model.py`:

```python
# More epochs for better quality
num_train_epochs=5

# Larger batch for faster training (needs more GPU memory)
per_device_train_batch_size=8

# Higher learning rate for faster convergence
learning_rate=3e-4
```

### Different Base Models

Try other base models:

```python
# Smaller, faster
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Larger, better quality
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Code-specialized
BASE_MODEL = "codellama/CodeLlama-7b-hf"
```

## IDE Integration

### VS Code Extension

```javascript
const { exec } = require('child_process');

function getFIMCompletion(prefix, suffix) {
    const prompt = `<|fim_prefix|>${prefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
    
    return new Promise((resolve, reject) => {
        exec(
            `ollama run fim-code-completion "${prompt}"`,
            (error, stdout, stderr) => {
                if (error) reject(error);
                else resolve(stdout.trim());
            }
        );
    });
}
```

### API Server

```python
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    
    result = subprocess.run(
        ['ollama', 'run', 'fim-code-completion', prompt],
        capture_output=True,
        text=True
    )
    
    return jsonify({
        'completion': result.stdout.strip(),
        'success': result.returncode == 0
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## Performance

### Inference Speed

| Hardware | Tokens/sec | Latency |
|----------|-----------|---------|
| CPU (8 cores) | ~15-20 | ~500ms |
| M1 Mac | ~30-40 | ~250ms |
| NVIDIA RTX 3090 | ~80-100 | ~100ms |

### Model Size

| Format | Size | Quality |
|--------|------|---------|
| Full FP16 | ~2.4 GB | 100% |
| Q4_0 | ~1.4 GB | 98% |
| Q4_K_M | ~1.5 GB | 99% |
| Q2_K | ~0.9 GB | 93% |

## Troubleshooting

### Out of Memory

If training fails with OOM:

```python
# Reduce batch size
per_device_train_batch_size=2
gradient_accumulation_steps=4

# Enable gradient checkpointing
gradient_checkpointing=True

# Use 8-bit quantization
load_in_8bit=True
```

### Poor Completion Quality

1. **More training data**: Generate 5000+ examples
2. **More epochs**: Train for 5-10 epochs
3. **Better base model**: Use CodeLlama or larger Llama
4. **Add your domain**: Include your codebase in training

### Slow Inference

1. **Better quantization**: Use Q4_K_M instead of Q4_0
2. **Batch prompts**: Process multiple completions together
3. **GPU acceleration**: Use CUDA/Metal backend
4. **Model size**: Try TinyLlama for faster inference

## Best Practices

### Training
- ✅ Include diverse code patterns
- ✅ Mix languages your IDE targets
- ✅ Add real-world examples from your projects
- ✅ Validate on held-out test set

### Deployment
- ✅ Cache model in memory
- ✅ Use streaming for long completions
- ✅ Set timeout for responsiveness
- ✅ Fallback to simpler completion

### Quality
- ✅ Filter training data quality
- ✅ Remove duplicates
- ✅ Balance languages
- ✅ Test edge cases

## Next Steps

1. **Expand training data**: Add more languages (Rust, Go, TypeScript)
2. **Multi-line completion**: Train on larger context windows
3. **Function-level completion**: Train on whole function bodies
4. **Docstring generation**: Add doc-to-code examples
5. **Test generation**: Train on test/implementation pairs

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Ollama Documentation](https://ollama.ai/docs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [FIM Training Paper](https://arxiv.org/abs/2207.14255)

## License

The FIM training code is MIT licensed. Trained models inherit the license of the base model (Llama 3.2: Llama 3 Community License).

---

**Happy Coding with FIM! 🚀**

For questions or issues, open a GitHub issue or check the documentation.
