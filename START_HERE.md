# 🚀 START HERE - JARVIS AI

Welcome to JARVIS AI! This is the quickest way to get started.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies (First Time Only)

```bash
# Linux/Mac
./install_prerequisites.sh

# Windows - Download and install:
# - Node.js from https://nodejs.org
# - Python from https://python.org
# - Ollama from https://ollama.ai/download (optional)
```

### Step 2: Run JARVIS

```bash
# Linux/Mac
./run_ai.sh

# Windows
run_ai.bat
```

### Step 3: Start Chatting!

Open your browser to: **http://localhost:3001**

---

## 🎯 What Happens Next?

1. **Menu appears** - Choose your backend:
   - **Ollama** (easiest) - Many pre-trained models
   - **Pinokio** - GUI for model management
   - **Local** - Use your own trained models

2. **Auto-setup** - Script configures everything

3. **Open browser** - Go to http://localhost:3001

4. **Start chatting!** - That's it!

---

## 🤖 Want to Use Ollama? (Recommended)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a model
ollama pull llama3.2

# 3. Run JARVIS
./run_ai.sh

# 4. Select "Ollama" from menu
# 5. Enter model name: llama3.2
```

---

## 📋 What Do You Need?

### Minimum Requirements
- **Node.js** 16+ (Required)
- **Python** 3.8+ (Required for local inference)
- **8GB RAM** (16GB recommended)
- **10GB free space**

### Optional
- **Ollama** - For easy model management
- **NVIDIA GPU** - For faster inference

---

## 🎨 The New Web Interface

Once running, you'll see a beautiful UI with:

- 🔗 **Backend Selector** - Choose Ollama, Pinokio, or Local
- 📊 **Connection Status** - Visual indicator
- 💬 **Real-time Chat** - Streaming responses
- 📈 **Statistics** - Track messages, response time, tokens
- 📱 **Responsive Design** - Works on any device

---

## 🆘 Need Help?

### "Ollama not detected"
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### "Node.js not found"
```bash
# Download from https://nodejs.org
# Or install via package manager
```

### "Python not found"
```bash
# Download from https://python.org
# Or install via package manager
```

### Port already in use?
```bash
# Kill existing process
pkill -f "node server.js"
# Then try again
./run_ai.sh
```

### Check logs
```bash
cat logs/server.log
cat logs/inference.log
```

---

## 📚 Learn More

- **[Quick Start Guide](README_EASY_RUN.md)** - Complete easy-run guide
- **[Installation Guide](INSTALL_GUIDE.md)** - Detailed setup instructions
- **[Quick Reference](QUICK_REFERENCE.md)** - Commands and troubleshooting
- **[Features](FILES_AND_FEATURES.md)** - What's new and how it works

---

## 🎉 You're Ready!

Just run:

```bash
./run_ai.sh
```

Or on Windows:

```cmd
run_ai.bat
```

**That's literally all you need to do!** 🎊

---

**Made with ❤️ to make AI accessible to everyone**
