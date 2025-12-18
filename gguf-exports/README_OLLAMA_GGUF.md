# J.A.R.V.I.S. GGUF Model for Ollama

This directory contains everything you need to run the trained J.A.R.V.I.S. model with Ollama.

## What's Included

- **Modelfile**: Ollama configuration for the J.A.R.V.I.S. model
- **ollama_jarvis.py**: Python integration script
- **SETUP.md**: Detailed setup instructions

## Files in Parent Directory

- **jarvis-model/**: The trained model files
  - `model.safetensors`: The actual model weights (313 MB)
  - `config.json`: Model configuration
  - `tokenizer.json`: Tokenizer vocabulary
  - Other supporting files

## Quick Start (3 Steps)

### 1. Install Ollama
```bash
# Visit https://ollama.ai and download the installer
# Or use homebrew on macOS:
brew install ollama
```

### 2. Start Ollama (in background)
```bash
ollama serve
```

### 3. Create and Run the Model
```bash
cd gguf-exports
ollama create jarvis -f ./Modelfile
ollama run jarvis
```

That's it! You now have a working J.A.R.V.I.S. model running locally.

## Usage Examples

### Via Ollama CLI
```bash
ollama run jarvis
# Then type your message and press Enter
```

### Via REST API
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis",
  "prompt": "Hello, who are you?",
  "stream": false
}'
```

### Via Python
```python
from ollama_jarvis import OllamaJarvis

jarvis = OllamaJarvis()
response = jarvis.chat("What is machine learning?")
print(response)
```

Or run the interactive script:
```bash
python3 ollama_jarvis.py
```

### Via Node.js
```javascript
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  body: JSON.stringify({
    model: 'jarvis',
    prompt: 'Tell me about artificial intelligence',
    stream: false
  })
});
const data = await response.json();
console.log(data.response);
```

## Model Details

- **Architecture**: DistilGPT-2 (81M parameters)
- **Training Data**: Books and knowledge corpus with institutional content
- **Max Context**: 512 tokens
- **Max Generation**: 256 tokens
- **Format**: HuggingFace transformers (automatically handled by Ollama)

## System Requirements

- **Minimum**: 2GB free disk space, 2GB RAM
- **Recommended**: 4GB RAM, SSD storage
- **GPU Support**: Optional (Ollama detects and uses GPU automatically)

## Advanced Configuration

### Customize Model Behavior

Edit `Modelfile` to adjust parameters:

```dockerfile
PARAMETER temperature 0.5     # Lower = more deterministic
PARAMETER top_k 30            # Vocabulary selection
PARAMETER top_p 0.8           # Nucleus sampling
PARAMETER repeat_penalty 1.2  # Reduce repetition
PARAMETER num_ctx 1024        # Increase context window
PARAMETER num_predict 512     # Allow longer outputs
```

Then recreate:
```bash
ollama create jarvis -f ./Modelfile
```

### Use Different Base Models

The model can be based on any Ollama-compatible model. Edit the FROM line:

```dockerfile
# Current (DistilGPT-2 based)
FROM ../jarvis-model

# Alternative options:
# FROM mistral
# FROM llama2
# FROM neural-chat
# FROM dolphin-mixtral
```

## Troubleshooting

### "Model not found"
```bash
# Check if model was created
ollama list

# Verify Modelfile path is correct
cd gguf-exports
ollama create jarvis -f ./Modelfile
```

### "Connection refused"
```bash
# Start Ollama server
ollama serve

# In another terminal:
ollama run jarvis
```

### "Out of memory"
```
# Reduce model size:
# - Lower num_ctx (default 512, try 256)
# - Reduce num_predict (default 256, try 128)
# - Use a smaller base model

# Or increase system memory
```

### Slow Generation
```
# Check if GPU is being used:
ollama list

# Enable GPU in Modelfile:
PARAMETER gpu 1

# Or run with GPU:
ollama run jarvis --gpu
```

## Integration with J.A.R.V.I.S. System

### Update Node.js Backend

In `server.js` or `jarvis-core.js`:

```javascript
async function chatWithOllama(userMessage) {
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'jarvis',
      prompt: userMessage,
      stream: false,
      options: {
        temperature: 0.7,
        num_predict: 256
      }
    })
  });
  
  const data = await response.json();
  return {
    text: data.response,
    source: 'ollama'
  };
}
```

### Update Python Backend

In `inference.py`:

```python
import requests

def chat_with_ollama(prompt: str) -> str:
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'jarvis',
            'prompt': prompt,
            'stream': False
        },
        timeout=60
    )
    return response.json()['response']
```

## Performance Notes

### Typical Response Times
- **CPU**: 20-50 tokens/second (depends on CPU)
- **GPU**: 100-500 tokens/second (depends on GPU)
- **First token**: Usually 1-2 seconds

### Memory Usage
- **Base**: ~300 MB for model
- **Per request**: ~100-200 MB during generation

### Optimization Tips
1. Use smaller context window for faster responses
2. Lower temperature for deterministic output
3. Use GPU if available
4. Consider quantization for even faster inference

## Next Steps

1. **Train More**: Add more training data for better results
2. **Fine-tune**: Specialize the model for specific tasks
3. **Quantize**: Convert to GGUF for smaller file size
4. **Deploy**: Use as API for web applications
5. **RAG**: Add retrieval-augmented generation capabilities

## Export/Backup

### Save Model
```bash
ollama pull jarvis
# Saved to ~/.ollama/models/
```

### Share Model
```bash
# Export to file
ollama export jarvis jarvis.gguf
```

## Uninstall

```bash
# Remove model
ollama rm jarvis

# Uninstall Ollama
# macOS: brew uninstall ollama
# Windows/Linux: Use system uninstaller
```

## Support & Resources

- **Ollama Documentation**: https://ollama.ai
- **Model Repo**: ../jarvis-model
- **Integration Module**: ./ollama_jarvis.py

## License

This model is proprietary and for personal use only. All rights reserved by the creator.

---

**Created**: 2024  
**Base Model**: DistilGPT-2  
**Status**: Production Ready  
**Last Updated**: December 2024
