# J.A.R.V.I.S. + Ollama Setup Guide

## Prerequisites
1. Install Ollama: https://ollama.ai
2. Ensure Python 3.8+ is available
3. Install requests: `pip install requests`

## Setup Steps

### 1. Download Ollama
- Visit https://ollama.ai
- Download the appropriate version for your OS (Windows, macOS, Linux)
- Install and run Ollama

### 2. Verify Ollama is Running
```bash
curl http://localhost:11434/api/tags
```

### 3. Create the J.A.R.V.I.S. Model
```bash
cd ./gguf-exports
ollama create jarvis -f ./Modelfile
```

### 4. Test the Model
```bash
ollama run jarvis

# In the prompt, try:
# > Who are you?
# > Tell me about artificial intelligence
# > Help me with Python coding
```

### 5. Use via Python API
```bash
python3 ollama_jarvis.py
```

### 6. Use via cURL
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

## Integration with Node.js

Update your `server.js` or inference API to use Ollama:

```javascript
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  body: JSON.stringify({
    model: 'jarvis',
    prompt: userMessage,
    stream: false
  })
});
const data = await response.json();
return data.response;
```

## Performance Tips

- GPU: Ollama automatically uses GPU if available
- Temperature: Lower (0.3) for factual, Higher (0.9) for creative
- Context Size: Can be increased in Modelfile (num_ctx)
- Generation Length: Adjust num_predict parameter

## Troubleshooting

- **Connection refused**: Start Ollama with `ollama serve`
- **Model not found**: Verify creation with `ollama list`
- **Out of memory**: Reduce context size or batch size
- **Slow generation**: Check if GPU is being used

## Advanced Usage

### Customize Model Parameters
Edit `./Modelfile` and recreate:
```bash
ollama create jarvis -f ./Modelfile
```

### Use Different Base Models
Change the FROM line in Modelfile to other Ollama models:
- `FROM mistral` - Mistral 7B
- `FROM neural-chat` - Neural Chat
- `FROM dolphin-mixtral` - Dolphin Mixtral

### Export Model
```bash
ollama pull jarvis
# Model saved to ~/.ollama/models
```

## Next Steps

1. Fine-tune model with custom data
2. Create different model variants for different tasks
3. Deploy as API service
4. Integrate with web applications
5. Add RAG (Retrieval Augmented Generation) capabilities

---
For more information, visit: https://ollama.ai
