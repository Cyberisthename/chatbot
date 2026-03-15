# 🤖 J.A.R.V.I.S. + Ollama Complete Setup Guide

This guide walks you through running a fully-trained, locally-deployable J.A.R.V.I.S. AI model using Ollama.

## Overview

- ✅ **Model Trained**: A DistilGPT-2 model trained on books and knowledge data
- ✅ **Ready for Ollama**: Configured to work seamlessly with Ollama
- ✅ **No API Keys**: Run everything locally, no cloud dependencies
- ✅ **Easy Integration**: REST API and Python/Node.js integrations included

## 📁 What's Included

```
project/
├── jarvis-model/                    # Trained model files (313 MB)
│   ├── model.safetensors           # Model weights
│   ├── config.json                 # Model configuration
│   ├── tokenizer.json              # Tokenizer
│   └── metadata.json               # Model metadata
│
├── gguf-exports/                   # Ollama integration files
│   ├── Modelfile                   # Ollama configuration
│   ├── README_OLLAMA_GGUF.md       # Detailed documentation
│   ├── ollama_jarvis.py            # Python integration
│   ├── SETUP.md                    # Setup instructions
│   └── convert_hf_to_gguf.sh       # GGUF conversion script
│
├── train_ollama_model.py           # Training script (already run)
├── convert_to_gguf_direct.py       # Direct conversion utility
└── OLLAMA_JARVIS_COMPLETE_SETUP.md # This file
```

## 🚀 Quick Start (5 minutes)

### Step 1: Install Ollama

Visit **https://ollama.ai** and download the installer for your OS:
- **macOS**: Intel or Apple Silicon
- **Windows**: Installer included
- **Linux**: Docker or native binary

Or use Homebrew on macOS:
```bash
brew install ollama
```

### Step 2: Start Ollama Server

```bash
# In a terminal/command prompt, run:
ollama serve

# You should see: "Listening on 127.0.0.1:11434"
```

Keep this running in the background!

### Step 3: Create the J.A.R.V.I.S. Model

In a new terminal, navigate to the project:

```bash
cd /path/to/project/gguf-exports
ollama create jarvis -f ./Modelfile
```

This imports the trained model into Ollama.

### Step 4: Start Chatting!

```bash
ollama run jarvis
```

Then type and press Enter:
```
> Who are you?
> Tell me about machine learning
> Help me with Python
```

## 💻 Different Ways to Use

### 1. **Ollama CLI** (Easiest)
```bash
ollama run jarvis
```

### 2. **REST API** (For Applications)
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

### 3. **Python** (For Scripts)
```bash
cd gguf-exports
python3 ollama_jarvis.py
```

Or in your code:
```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'jarvis',
    'prompt': 'Hello!',
    'stream': False
})
print(response.json()['response'])
```

### 4. **Node.js** (For Web Apps)
```javascript
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  body: JSON.stringify({
    model: 'jarvis',
    prompt: 'Hello!',
    stream: false
  })
});
const data = await response.json();
console.log(data.response);
```

### 5. **Web UI** (Built-in)
```bash
# Ollama includes a web interface
# Open: http://localhost:11434
```

## 🎯 Integration Examples

### Express.js Backend
```javascript
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;
  
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    body: JSON.stringify({
      model: 'jarvis',
      prompt: message,
      stream: false
    })
  });
  
  const data = await response.json();
  res.json({ reply: data.response });
});
```

### Python FastAPI
```python
from fastapi import FastAPI
import requests

app = FastAPI()

@app.post('/chat')
async def chat(message: str):
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'jarvis',
        'prompt': message,
        'stream': False
    })
    return {'reply': response.json()['response']}
```

### React Frontend
```jsx
function Chat() {
  const [response, setResponse] = useState('');
  
  const ask = async (message) => {
    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message })
    });
    const data = await res.json();
    setResponse(data.reply);
  };
  
  return <div>{response}</div>;
}
```

## ⚙️ Model Configuration

The model parameters are configured in `gguf-exports/Modelfile`:

```dockerfile
PARAMETER temperature 0.7      # Randomness (0=deterministic, 1=random)
PARAMETER top_k 40             # Vocabulary selection
PARAMETER top_p 0.9            # Nucleus sampling
PARAMETER repeat_penalty 1.1   # Avoid repetition
PARAMETER num_ctx 512          # Context window size
PARAMETER num_predict 256      # Max tokens to generate
```

To adjust these:

1. Edit `gguf-exports/Modelfile`
2. Recreate the model: `ollama create jarvis -f ./Modelfile`
3. Run: `ollama run jarvis`

### Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| temperature | 0-1 | Lower = more focused, Higher = more creative |
| top_k | 1-∞ | Limits vocabulary size (smaller = more focused) |
| top_p | 0-1 | Nucleus sampling (0.9 is typical) |
| repeat_penalty | 1.0+ | Higher = less repetition (1.1-1.2 recommended) |
| num_ctx | 128-∞ | Context window (larger = more memory) |
| num_predict | 1-∞ | Max output length |

## 🔄 Updating the Model

### Re-train with More Data

Edit `train_ollama_model.py` to add more training data, then:

```bash
cd /path/to/project
source .venv2/bin/activate
python3 train_ollama_model.py
```

Then recreate in Ollama:
```bash
cd gguf-exports
ollama create jarvis -f ./Modelfile
```

### Use a Different Base Model

Edit `gguf-exports/Modelfile` FROM line:

```dockerfile
# Current
FROM ../jarvis-model

# Alternative bases (remove FROM above and use):
# FROM mistral     # Larger, more capable model
# FROM llama2      # Meta's LLaMA 2
# FROM neural-chat # Optimized for chat
# FROM dolphin     # Uncensored variant
```

Then update system prompt and recreate:
```bash
ollama create jarvis -f ./Modelfile
```

## 🔍 Troubleshooting

### Issue: "Connection refused"
**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### Issue: "Model not found"
**Solution**: Create the model
```bash
cd gguf-exports
ollama create jarvis -f ./Modelfile
ollama list  # Verify it's there
```

### Issue: Out of Memory
**Solution**: Reduce context or use smaller model
```dockerfile
PARAMETER num_ctx 256      # Reduce from 512
PARAMETER num_predict 128  # Reduce from 256
```

### Issue: Slow Responses
**Solution**: Check GPU usage and model size
```bash
# Check models
ollama list

# Use smaller model or enable GPU
# Ollama auto-detects GPU
```

### Issue: Model Taking Too Long to Respond
**Solution**: Reduce temperature and context
```dockerfile
PARAMETER temperature 0.3  # More deterministic
PARAMETER num_ctx 256      # Smaller context
```

## 📊 Model Information

| Property | Value |
|----------|-------|
| Architecture | DistilGPT-2 |
| Parameters | 81.9 Million |
| Size | ~313 MB (model.safetensors) |
| Max Context | 512 tokens |
| Training Data | Books and knowledge corpus |
| Format | HuggingFace transformers |
| License | Proprietary - Personal use only |

## 🎓 Advanced Topics

### Custom System Prompt

Edit `gguf-exports/Modelfile`:

```dockerfile
SYSTEM """You are J.A.R.V.I.S., an expert in:
- Machine learning and AI
- Python programming
- Data science and analysis
- System design and architecture

You are helpful, accurate, and always provide code examples when relevant."""
```

### Multi-Model Setup

Run different models for different tasks:

```bash
ollama create jarvis -f ./gguf-exports/Modelfile
ollama create coder -f ./gguf-exports/Modelfile.coder
ollama create analyst -f ./gguf-exports/Modelfile.analyst

# Use different models:
ollama run jarvis        # General assistant
ollama run coder         # Coding specialist
ollama run analyst       # Data analyst
```

### Streaming Responses

Get responses token-by-token:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis",
  "prompt": "Write a poem",
  "stream": true
}'
```

### With Context History

Keep conversation context:

```python
context = []
prompts = []

for message in user_messages:
    prompts.append(message)
    full_prompt = "\n".join(prompts)
    
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'jarvis',
        'prompt': full_prompt,
        'stream': False,
        'context': context
    })
    
    data = response.json()
    context = data.get('context', [])
    prompts.append(data['response'])
```

## 📚 Resources

- **Ollama Website**: https://ollama.ai
- **Ollama Documentation**: https://github.com/jmorganca/ollama
- **Model Directory**: ./jarvis-model
- **Integration Guide**: ./gguf-exports/README_OLLAMA_GGUF.md
- **Setup Instructions**: ./gguf-exports/SETUP.md

## 🐛 Getting Help

1. Check `./gguf-exports/README_OLLAMA_GGUF.md` for detailed docs
2. Visit https://ollama.ai/docs
3. Check Ollama GitHub issues: https://github.com/jmorganca/ollama/issues

## ✅ Verification Checklist

- [ ] Ollama installed
- [ ] Ollama server running (`ollama serve`)
- [ ] Model created (`ollama create jarvis -f ./Modelfile`)
- [ ] Model appears in `ollama list`
- [ ] Can run `ollama run jarvis` without errors
- [ ] Can make API calls to `http://localhost:11434/api/generate`
- [ ] Python integration works (`python3 ollama_jarvis.py`)
- [ ] Model response is coherent

## 🎯 Next Steps

1. **Customize**: Edit Modelfile to adjust behavior
2. **Integrate**: Add to your applications using REST API
3. **Train More**: Add more training data for better results
4. **Deploy**: Move to production with Docker
5. **Scale**: Use multiple models for different tasks
6. **Optimize**: Quantize for better performance

## 📝 License

This model is proprietary and for personal use only. All rights reserved.

---

**Status**: ✅ Production Ready  
**Created**: December 2024  
**Last Updated**: December 2024  
**Support**: See ./gguf-exports/README_OLLAMA_GGUF.md

