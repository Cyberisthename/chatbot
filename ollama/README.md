# JARVIS-2v Ollama Integration

This directory contains the Ollama Modelfile for running JARVIS-2v with Ollama.

## Prerequisites

1. Install Ollama: https://ollama.ai
2. Ensure you have a GGUF model file in `./models/` directory

## Setup Steps

### 1. Update Model Path

Edit `Modelfile` and update the `FROM` line to point to your GGUF model:

```
FROM ./models/jarvis-7b-q4_0.gguf
```

Or use an absolute path:

```
FROM /path/to/your/model.gguf
```

### 2. Create Ollama Model

From the project root directory:

```bash
cd /path/to/jarvis-2v
ollama create jarvis2v -f ollama/Modelfile
```

This will register the model with Ollama.

### 3. Run JARVIS-2v

```bash
ollama run jarvis2v
```

This starts an interactive chat session with JARVIS-2v.

### 4. Use with API

The model is now available via Ollama's API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis2v",
  "prompt": "Hello JARVIS, who created you?"
}'
```

## Integration with Adapter System

JARVIS-2v's intelligence doesn't come from the base model alone. The model works with:

1. **Adapter Engine** - Routes queries through Y/Z/X bit system
2. **Memory System** - Persists facts and conversation history
3. **Quantum Lab** - Provides experimental context
4. **Knowledge Graph** - Connects related adapters

To get the full JARVIS-2v experience, you should:

1. Train adapters using the training scripts:
   ```bash
   python scripts/train_adapters.py --input data/raw
   python scripts/train_idi_stream.py --max-books 100
   ```

2. Run the full JARVIS-2v server (which integrates everything):
   ```bash
   python inference.py models/jarvis-7b-q4_0.gguf --port 8000
   ```

3. Or use the Node.js server with web UI:
   ```bash
   npm start
   ```

## Customization

You can customize the Modelfile:

- **Temperature**: Control randomness (0.0 = deterministic, 1.0 = creative)
- **Context Size**: Increase `num_ctx` for longer conversations
- **System Prompt**: Modify the SYSTEM section to change personality
- **Stop Sequences**: Add custom stop tokens

## Model Parameters Explained

- `temperature`: 0.7 - Balanced creativity and consistency
- `top_p`: 0.9 - Nucleus sampling for diverse responses
- `top_k`: 40 - Limits token selection pool
- `num_ctx`: 2048 - Context window size
- `repeat_penalty`: 1.1 - Reduces repetition

## Updating the Model

If you retrain or get a new GGUF file:

```bash
# Remove old model
ollama rm jarvis2v

# Create new model
ollama create jarvis2v -f ollama/Modelfile
```

## Using Different Model Sizes

If you have multiple GGUF variants (e.g., Q4, Q5, Q8):

```bash
# Create different variants
ollama create jarvis2v-q4 -f ollama/Modelfile-q4
ollama create jarvis2v-q8 -f ollama/Modelfile-q8

# Run specific variant
ollama run jarvis2v-q8
```

## Troubleshooting

### "Model not found" error

Make sure the path in the Modelfile is correct and the GGUF file exists:

```bash
ls -lh models/*.gguf
```

### "Out of memory" error

Reduce context size or use a smaller quantization:

```
PARAMETER num_ctx 1024
```

### Slow inference

Use GPU acceleration if available:

```bash
# Check GPU availability
ollama list

# Ollama automatically uses GPU if detected
```

## Learn More

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [JARVIS-2v Architecture](../README.md)
