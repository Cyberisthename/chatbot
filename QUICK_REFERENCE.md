# 🎯 JARVIS AI - Quick Reference Card

> Print this or keep it handy for quick commands!

## 🚀 Start JARVIS

### The One Command You Need:

```bash
./run_ai.sh              # Linux/Mac/WSL
run_ai.bat              # Windows
```

That's it! Everything else is automatic.

## 📋 Installation Commands

```bash
# Auto-install everything
./install_prerequisites.sh    # Linux/Mac

# Manual install
npm install                    # Node dependencies
pip3 install -r requirements.txt  # Python dependencies
```

## 🤖 Backend Commands

### Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama service
ollama serve

# List models
ollama list

# Run a model
ollama run llama3.2
```

### Local Inference
```bash
# Train your model
python train_jarvis.py

# Export to GGUF
python train_and_export_gguf.py

# Start inference backend
python inference.py
```

## 🌐 Server Commands

```bash
# Start server
node server.js

# Start with custom port
PORT=3002 node server.js

# Start in development mode
npm run dev

# Start all services
./start.sh
```

## 📊 Model Management

```bash
# List available models
ollama list

# Show model info
ollama show llama3.2

# Download model
ollama pull mistral

# Remove model
ollama rm llama3.2

# Your models are in: models/
```

## 🛠️ Configuration

### Edit `config.yaml`:
```yaml
model:
  path: "./models/jarvis-7b-q4_0.gguf"
  temperature: 0.7
  context_size: 2048

api:
  host: "0.0.0.0"
  port: 3001
```

## 🔍 Troubleshooting

```bash
# View server logs
cat logs/server.log

# View inference logs
cat logs/inference.log

# Restart if stuck
pkill -f "node server.js"
./run_ai.sh

# Check if port is in use
lsof -ti:3001  # Linux/Mac
netstat -ano | findstr :3001  # Windows

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

## 📡 API Endpoints

```bash
# Health check
curl http://localhost:3001/api/health

# Chat
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Model info
curl http://localhost:3001/api/model

# System status
curl http://localhost:3001/api/status
```

## 🖥️ Web Interface

```
URL: http://localhost:3001
    http://localhost:3001/local_ai_ui.html

Features:
- Backend selector (Ollama/Pinokio/Local)
- Real-time chat
- Statistics panel
- Connection status
```

## 📝 Common Tasks

### Start Fresh
```bash
# 1. Kill all processes
pkill -f "node server.js"
pkill -f "python inference.py"

# 2. Clean up
rm -rf logs/*

# 3. Start again
./run_ai.sh
```

### Use Different Model
```bash
# 1. Pull new model
ollama pull mistral

# 2. Run script
./run_ai.sh

# 3. Enter model name when prompted
```

### Test Installation
```bash
# Test Node.js
node --version

# Test Python
python3 --version

# Test Ollama
ollama --version

# Test server
curl http://localhost:3001/api/health
```

## 📁 Directory Structure

```
project/
├── run_ai.sh              # Main run script ⭐
├── local_ai_ui.html       # Web UI ⭐
├── server.js              # Node.js server
├── inference.py           # Python backend
├── config.yaml            # Configuration
├── models/                # GGUF models
├── adapters/              # Learned adapters
├── logs/                  # Runtime logs
└── requirements.txt       # Python deps
```

## 🎯 Quick Decision Tree

**What do you want to do?**

┌─────────────────────────────────────────┐
│ I want to start JARVIS                  │
└─────────────────┬───────────────────────┘
                  │
            ┌─────▼─────┐
            │./run_ai.sh│
            └───────────┘

┌─────────────────────────────────────────┐
│ I want to try different models           │
└─────────────────┬───────────────────────┘
                  │
            ┌─────▼─────┐
            │ollama pull│
            │<model-name>│
            └───────────┘

┌─────────────────────────────────────────┐
│ I want to use my own model              │
└─────────────────┬───────────────────────┘
                  │
            ┌─────▼─────┐
            │Copy to:   │
            │models/    │
            └───────────┘

┌─────────────────────────────────────────┐
│ Something is wrong                      │
└─────────────────┬───────────────────────┘
                  │
            ┌─────▼─────┐
            │Check logs:│
            │logs/*.log │
            └───────────┘

## 🔑 Essential Files

- `run_ai.sh` - Start everything ⭐⭐⭐
- `local_ai_ui.html` - Beautiful UI ⭐⭐
- `config.yaml` - Settings ⭐
- `logs/server.log` - Server logs
- `logs/inference.log` - AI logs

## 📞 Quick Help

1. **Can't start?** → Check logs in `logs/`
2. **Port error?** → Kill process with `pkill -f "node server.js"`
3. **Model missing?** → Run `ollama pull <model>`
4. **Update needed?** → `git pull` then `npm install`

## ✅ Installation Status

Check if everything is ready:

```bash
# Node.js?
node --version    # Should show v16+

# Python?
python3 --version # Should show Python 3.8+

# Ollama (optional)?
ollama --version  # Should show version

# Dependencies?
ls node_modules   # Should show many packages

# Models?
ls models/        # Should show .gguf files
```

## 🎉 You're All Set!

Just remember:

```bash
./run_ai.sh
```

Everything else is handled automatically!

---

**Save this file for quick reference! 📋**
