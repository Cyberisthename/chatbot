# 🎯 Easy Run Your JARVIS AI - One Script to Rule Them All!

> **TL;DR:** Just run `./run_ai.sh` (Linux/Mac) or `run_ai.bat` (Windows) and you're done!

## 🚀 What This Does

This script automates everything you need to run JARVIS:

✅ **Detects** available AI backends (Ollama, Pinokio, Local)
✅ **Configures** everything automatically
✅ **Downloads/Installs** any missing dependencies
✅ **Launches** the beautiful web UI
✅ **Manages** all services in one place

## 📥 Quick Start

### For Linux/Mac Users:

```bash
# Make it executable (first time only)
chmod +x run_ai.sh

# Run it!
./run_ai.sh
```

### For Windows Users:

```cmd
# Just double-click run_ai.bat
# or run from command prompt:
run_ai.bat
```

### What Happens Next:

1. **Menu appears** - Choose your backend (Ollama, Pinokio, or Local)
2. **Auto-setup** - Script configures everything for you
3. **Browser opens** - Or go to http://localhost:3001
4. **Start chatting!** - That's it!

## 🎮 Choosing Your Backend

### Option 1: Ollama (⭐ Easiest)

**Best for:** Beginners, quick setup, trying out AI

```bash
# The script will guide you through:
1. Select "Ollama" from menu
2. Enter model name (e.g., llama3.2, mistral)
3. Done!
```

**Why choose Ollama:**
- ✅ One-line install
- ✅ Many free models
- ✅ Good performance
- ✅ Active community

### Option 2: Pinokio (🎨 GUI Lovers)

**Best for:** People who prefer graphical interfaces

```bash
# The script will:
1. Select "Pinokio" from menu
2. Start Pinokio service
3. Open Pinokio interface
```

**Why choose Pinokio:**
- ✅ Beautiful GUI
- ✅ One-click model installs
- ✅ Easy model management
- ✅ Great for beginners

### Option 3: Local Inference (🔧 Power Users)

**Best for:** Using your own trained models, maximum control

```bash
# The script will:
1. Select "Local" from menu
2. Start Python backend
3. Load your GGUF models
```

**Why choose Local:**
- ✅ Use your own models
- ✅ Maximum control
- ✅ Works offline completely
- ✅ Best for trained JARVIS models

## 🖥️ The Web Interface

Once running, you get a **beautiful, modern UI** with:

### Connection Panel
- 🔗 Backend selector (Ollama/Pinokio/Local)
- 📊 Connection status indicator
- ⚙️ Model configuration
- 🌐 API URL settings

### Chat Interface
- 💬 Real-time chat with your AI
- ⌨️ Typing indicators
- 📜 Message history
- 🎨 Modern gradient design

### Statistics
- 📈 Message count
- ⏱️ Response times
- 🔢 Token usage
- ⏰ Session duration

## 📝 Example Usage

### Scenario 1: First Time Setup with Ollama

```bash
# 1. Install Ollama (if not already)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Run JARVIS
./run_ai.sh

# 3. Select option 1 (Ollama)
# 4. Enter model name: llama3.2
# 5. Open http://localhost:3001
# 6. Start chatting!
```

### Scenario 2: Using Your Trained Model

```bash
# 1. Train your model
python train_jarvis.py

# 2. Export to GGUF
python train_and_export_gguf.py

# 3. Run JARVIS
./run_ai.sh

# 4. Select option 3 (Local)
# 5. It auto-detects your model!
# 6. Start chatting with YOUR model
```

### Scenario 3: Quick Demo Without Models

```bash
# 1. Run script
./run_ai.sh

# 2. Select "Demo" mode
# 3. Open web interface
# 4. See the UI (AI responses will be simulated)
```

## 🛠️ What the Script Does Behind the Scenes

### Detection Phase
- Checks for Ollama installation
- Checks for Pinokio installation
- Checks for Python and Node.js
- Validates all dependencies

### Setup Phase
- Installs missing Node packages
- Installs missing Python packages
- Downloads/verifies models
- Updates configuration files

### Launch Phase
- Starts Python inference backend (if using local)
- Starts Node.js web server
- Opens web interface
- Displays connection details
- Manages all processes

### Cleanup Phase
- Stops all services on exit
- Closes all connections
- Cleans up temporary files

## 🔍 Troubleshooting

### "Ollama not detected"
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai/download
```

### "Python not found"
```bash
# On Ubuntu/Debian
sudo apt install python3 python3-pip

# On Mac
brew install python3

# On Windows
# Download from https://python.org
```

### "Node.js not found"
```bash
# On Ubuntu/Debian
sudo apt install nodejs npm

# On Mac
brew install node

# On Windows
# Download from https://nodejs.org
```

### Port already in use
```bash
# Kill existing process
pkill -f "node server.js"

# Or use a different port
PORT=3002 ./run_ai.sh
```

### Check logs
```bash
# Server logs
cat logs/server.log

# Inference logs
cat logs/inference.log
```

## 📊 Backend Comparison

| Feature | Ollama | Pinokio | Local |
|---------|--------|---------|-------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Model Variety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Custom Models** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GUI** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Offline** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Beginner Friendly** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🎯 Use Case Recommendations

### "I just want to try AI quickly"
→ **Use Ollama** - Fastest setup, many models

### "I prefer clicking over typing"
→ **Use Pinokio** - Beautiful GUI, easy model management

### "I trained my own model"
→ **Use Local** - Directly use your GGUF files

### "I need maximum control"
→ **Use Local** - Full configuration options

### "I'm offline"
→ **Use Ollama or Local** - Both work offline

### "I want the best performance"
→ **Use Local** - Direct model access, no overhead

## 📚 Next Steps

1. **Chat with your AI** - Start asking questions
2. **Train your model** - Use your own data
3. **Customize the UI** - Edit `local_ai_ui.html`
4. **Configure settings** - Edit `config.yaml`
5. **Deploy to web** - See `VERCEL_DEPLOYMENT.md`

## 🤝 Getting Help

1. Read this file carefully
2. Check `QUICKSTART_LOCAL_AI.md` for detailed guide
3. View logs in `logs/` directory
4. Check error messages
5. Verify all dependencies are installed

## 🎉 You're Ready!

Just remember one command:

**Linux/Mac:** `./run_ai.sh`
**Windows:** `run_ai.bat`

That's it! Your JARVIS AI is just one command away!

---

**Made with ❤️ to make AI accessible to everyone**
