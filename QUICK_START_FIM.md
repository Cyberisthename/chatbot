# Quick Start: FIM Code Completion Model

## What is this?

Train a **Fill-in-the-Middle (FIM)** code completion model that can intelligently complete code given both prefix and suffix context - perfect for IDE integrations!

## One-Command Setup

```bash
./train_fim_and_export.sh
```

This will:
1. Generate 2000+ training examples
2. Fine-tune Llama-3.2-1B with LoRA
3. Merge and export to GGUF (~1.4 GB)
4. Create ready-to-use Modelfile

## Install & Test

```bash
# Install in Ollama
ollama create fim-code-completion -f Modelfile.fim

# Quick test
ollama run fim-code-completion

# Run automated tests
python test_fim_model.py

# Interactive testing
python test_fim_model.py --interactive
```

## Usage Example

### Command Line

```bash
# Complete code with prefix and suffix
ollama run fim-code-completion "<|fim_prefix|>def sort_array(arr):<|fim_suffix|>    return sorted_arr<|fim_middle|>"
```

### Python API

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
prefix = "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    "
suffix = "\n        if arr[mid] == target:\n            return mid"
completion = complete_code(prefix, suffix)
print(completion)
```

## How It Works

FIM models understand code context from **both directions**:

```python
# PREFIX (before cursor)
def fibonacci(n):
    if n <= 1:
        return n
    
    [CURSOR]
    
# SUFFIX (after cursor)
    return fibonacci(n-1) + fibonacci(n-2)
```

The model generates: `# Recursive case follows`

## Training Data

Includes examples from:
- **Python**: algorithms, ML code, web scraping
- **JavaScript**: async patterns, React, utilities  
- **SQL**: complex queries, joins, aggregations

Each sample is split at random points to create realistic FIM scenarios.

## Customization

### Add Your Own Code

Edit `train_fim_model.py`:

```python
custom_samples = [
    {
        "code": """your code here""",
        "language": "python"
    }
]
```

### Adjust Training

```python
# More training data
num_samples=5000

# More epochs
num_train_epochs=5

# Larger model
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
```

## Model Specs

| Metric | Value |
|--------|-------|
| Base Model | Llama-3.2-1B-Instruct |
| Parameters | 1.2B |
| Training | LoRA (rank 16) |
| Quantization | Q4_0 (4-bit) |
| Size | ~1.4 GB |
| Speed | 15-20 tok/s (CPU) |

## Performance Tips

### Faster Inference
- Use GPU: `ollama run fim-code-completion --gpu`
- Smaller model: Try TinyLlama base
- Better quantization: Q4_K_M

### Better Quality
- More training data: 5000+ examples
- More epochs: 5-10
- Larger base model: Llama-3.2-3B or CodeLlama
- Add domain-specific code

## Troubleshooting

### Out of Memory
```python
# In train_fim_model.py
per_device_train_batch_size=2
gradient_accumulation_steps=4
gradient_checkpointing=True
```

### Model Not Found
```bash
# Check Ollama has the model
ollama list

# Recreate if needed
ollama create fim-code-completion -f Modelfile.fim
```

### Slow Training
- Use GPU if available
- Reduce training data: 500-1000 samples
- Use smaller base model: TinyLlama

## IDE Integration

### VS Code Example

```javascript
// extension.js
const vscode = require('vscode');
const { exec } = require('child_process');

function activate(context) {
    let provider = vscode.languages.registerCompletionItemProvider(
        'python',
        {
            provideCompletionItems(document, position) {
                const prefix = document.getText(
                    new vscode.Range(0, 0, position.line, position.character)
                );
                const suffix = document.getText(
                    new vscode.Range(position, document.lineCount, 0)
                );
                
                return getCompletion(prefix, suffix);
            }
        }
    );
    
    context.subscriptions.push(provider);
}

function getCompletion(prefix, suffix) {
    return new Promise((resolve) => {
        const prompt = `<|fim_prefix|>${prefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
        
        exec(
            `ollama run fim-code-completion "${prompt}"`,
            (error, stdout) => {
                if (!error) {
                    const item = new vscode.CompletionItem(
                        stdout.trim(),
                        vscode.CompletionItemKind.Snippet
                    );
                    resolve([item]);
                } else {
                    resolve([]);
                }
            }
        );
    });
}
```

## What's Next?

1. **Expand languages**: Add Rust, Go, TypeScript
2. **Multi-line completion**: Increase context window
3. **Function-level**: Train on full functions
4. **Test generation**: Add test/impl pairs
5. **Docstring gen**: Train doc-to-code

## Resources

- 📖 Full Guide: [FIM_MODEL_GUIDE.md](FIM_MODEL_GUIDE.md)
- 🔧 Training Script: [train_fim_model.py](train_fim_model.py)
- 🧪 Test Suite: [test_fim_model.py](test_fim_model.py)
- 🚀 Pipeline: [train_fim_and_export.sh](train_fim_and_export.sh)

## Support

For issues or questions:
1. Check [FIM_MODEL_GUIDE.md](FIM_MODEL_GUIDE.md)
2. Run `python test_fim_model.py` for diagnostics
3. Open a GitHub issue

---

**Happy Coding! 🚀**

Made with ❤️ for developers who want smart code completion.
