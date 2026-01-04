# 🎯 Getting Started with JARVIS AI - 2 Minute Guide

## 🚀 The Quickest Way to Start

```bash
./run_ai.sh
```

That's it! The script will:
1. Show you a menu of available backends
2. Set up everything automatically
3. Launch the web UI
4. Show you the URL to open

---

## 📖 What Do You Want to Do?

### I just want to try AI quickly

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Run JARVIS
./run_ai.sh

# 3. Select "Ollama" from menu
# 4. Enter: llama3.2
# 5. Open: http://localhost:3001
```

### I have a trained model I want to use

```bash
# 1. Place your GGUF model in models/ directory
cp my-model.gguf models/

# 2. Run JARVIS
./run_ai.sh

# 3. Select "Local" from menu
# 4. Model auto-detected!
# 5. Open: http://localhost:3001
```

### I want to see the beautiful UI

```bash
# 1. Run JARVIS
./run_ai.sh

# 2. Open browser to:
http://localhost:3001

# 3. Enjoy the modern interface!
```

---

## ⚡ Common Commands

### Start JARVIS
```bash
./run_ai.sh              # Linux/Mac
run_ai.bat              # Windows
```

### Install Everything (First Time)
```bash
./install_prerequisites.sh
```

### Pull an Ollama Model
```bash
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
```

### Stop JARVIS
```bash
# Press Ctrl+C in the terminal
```

---

## 🔧 Troubleshooting

### "Node.js not found"
Download from: https://nodejs.org

### "Python not found"
Download from: https://python.org

### "Ollama not detected"
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Port already in use?
```bash
pkill -f "node server.js"
./run_ai.sh
```

### View logs
```bash
cat logs/server.log
cat logs/inference.log
```

---

## 📚 More Documentation

- **Quick Start:** `README_EASY_RUN.md`
- **Detailed Guide:** `QUICKSTART_LOCAL_AI.md`
- **Installation:** `INSTALL_GUIDE.md`
- **Quick Reference:** `QUICK_REFERENCE.md`
- **Features:** `FILES_AND_FEATURES.md`
- **Full Summary:** `SUMMARY.md`

---

## 🎉 You're Ready!

Just run:

```bash
./run_ai.sh
```

And open: http://localhost:3001

**That's it!** 🎊
