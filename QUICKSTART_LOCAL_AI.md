# 🚀 JARVIS Local AI - Quick Start Guide

Run your JARVIS AI system with Ollama, Pinokio, or local inference!

## 📋 Prerequisites

### Option 1: Using Ollama (Recommended for beginners)
- Install Ollama: https://ollama.ai/download
- Or run: `curl -fsSL https://ollama.ai/install.sh | sh`

### Option 2: Using Pinokio
- Install Pinokio: https://pinokio.computer

### Option 3: Local Inference
- Python 3.8+
- Node.js 16+
- GGUF model file (optional - can train your own)

## ⚡ Quick Start

### The Easy Way - One Command!

```bash
./run_ai.sh
```

That's it! The script will:
1. ✅ Detect available AI backends (Ollama, Pinokio, Local)
2. 📋 Show you a menu of options
3. 🤖 Set up the selected backend automatically
4. 🌐 Launch the beautiful web UI at http://localhost:3001

### What Happens After Running

1. **Select your backend** from the menu
2. The script **automatically configures** everything
3. **Open your browser** to http://localhost:3001
4. **Start chatting!**

## 🎯 Backend Options

### 1. Ollama (Recommended)

**Pros:**
- ✅ Easiest to set up
- ✅ Many pre-trained models available
- ✅ Active community support
- ✅ Good performance on most hardware

**How it works:**
- Downloads and runs models locally
- No Python dependencies needed
- Supports various model sizes

**Example:**
```bash
# The script will ask for a model name
# Common models: llama3.2, mistral, codellama
```

### 2. Pinokio

**Pros:**
- ✅ Beautiful GUI for model management
- ✅ Easy to install and run models
- ✅ One-click model downloads
- ✅ Great for beginners

**How it works:**
- Runs as a local service
- Manages models automatically
- Provides web interface for model selection

### 3. Local Inference

**Pros:**
- ✅ Use your own trained models
- ✅ Maximum control
- ✅ No external dependencies
- ✅ Works offline completely

**How it works:**
- Uses your GGUF models directly
- Runs Python inference backend
- Best for custom-trained JARVIS models

## 🖥️ Web Interface Features

The new **Local AI UI** includes:

### Connection Panel
- **Backend Selection**: Choose Ollama, Pinokio, or Local
- **Model Selection**: Pick or specify your model
- **Connection Status**: Visual indicator (Connected/Disconnected)
- **API URL Configuration**: Customize backend URLs

### Chat Interface
- **Beautiful Design**: Modern gradient UI with glassmorphism
- **Real-time Responses**: Streaming responses as they generate
- **Typing Indicator**: Shows when AI is thinking
- **Message History**: Complete conversation history

### Statistics Panel
- **Message Count**: Track total messages exchanged
- **Response Time**: Average response time in seconds
- **Token Count**: Total tokens generated
- **Session Time**: Track how long you've been chatting

### Visual Features
- 🎨 Animated gradient headers
- 💫 Glowing effects and animations
- 🌙 Dark theme optimized for low-light
- 📱 Fully responsive design

## 📁 Project Structure

```
project/
├── run_ai.sh              # Main run script (THIS IS WHAT YOU USE)
├── local_ai_ui.html       # New beautiful UI
├── server.js              # Node.js web server
├── inference.py           # Python inference backend
├── config.yaml            # Configuration file
├── models/                # Place GGUF models here
│   └── *.gguf
├── adapters/              # Learned adapters
└── logs/                  # Runtime logs
```

## 🔧 Advanced Usage

### Using Specific Models

**With Ollama:**
```bash
# The script will prompt for model name
# Or pre-pull a model:
ollama pull llama3.2
ollama pull mistral
```

**With Local Inference:**
```bash
# Place your GGUF model in the models/ directory
cp /path/to/your-model.gguf models/
# The script will auto-detect it
```

### Custom Configuration

Edit `config.yaml`:
```yaml
model:
  path: "./models/your-model.gguf"
  context_size: 2048
  temperature: 0.7
  max_tokens: 2048
  gpu_layers: 30  # Set higher if you have GPU

api:
  host: "0.0.0.0"
  port: 3001
```

### Training Your Own Model

```bash
# Train a model on your data
python train_jarvis.py

# Export to GGUF format
python train_and_export_gguf.py

# The model will be in models/ directory
# Then run: ./run_ai.sh
```

### Running Without Script

If you want to run components manually:

```bash
# Start Python inference backend (for local models)
python inference.py &

# Start web server
node server.js &

# Open http://localhost:3001
```

## 🐛 Troubleshooting

### "Ollama not detected"
- Install Ollama: https://ollama.ai/download
- Make sure Ollama service is running: `ollama serve`

### "Pinokio not detected"
- Install Pinokio: https://pinokio.computer
- Start Pinokio application

### "Python not found"
- Install Python 3.8+: https://python.org/downloads
- On Ubuntu: `sudo apt install python3 python3-pip`

### "Node.js not found"
- Install Node.js: https://nodejs.org
- On Ubuntu: `sudo apt install nodejs npm`

### Model not loading
- Check the model file exists in `models/` directory
- Verify model format is GGUF
- Check logs in `logs/` directory

### Port already in use
- Kill existing process: `pkill -f "node server.js"`
- Or use different port: `PORT=3002 node server.js`

### View logs
```bash
# Server logs
cat logs/server.log

# Inference logs
cat logs/inference.log
```

## 📊 API Reference

### Health Check
```http
GET /api/health
```

### Chat
```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

### Model Info
```http
GET /api/model
```

### System Status
```http
GET /api/status
```

## 🎨 UI Customization

The UI is fully customizable! Edit `local_ai_ui.html` to:

- Change colors and gradients
- Modify layout
- Add new features
- Customize animations

## 🔒 Security Notes

- The server runs on localhost by default
- Change `host` in `config.yaml` if needed
- Use firewall rules for remote access
- Never expose the API publicly without authentication

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Pinokio Documentation](https://docs.pinokio.computer)
- [Main README](README.md)
- [Vercel Deployment Guide](VERCEL_DEPLOYMENT.md)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. View logs in the `logs/` directory
3. Review error messages carefully
4. Check if dependencies are installed

## 🎉 You're All Set!

Just run:
```bash
./run_ai.sh
```

And enjoy your JARVIS AI assistant!

---

**Made with ❤️ for local AI enthusiasts**
