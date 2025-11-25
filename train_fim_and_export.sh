#!/usr/bin/env bash
set -euo pipefail

# FIM Model Training and Export Pipeline
# This script trains a Fill-in-the-Middle code completion model and exports to GGUF

echo "======================================================================"
echo "🚀 Fill-in-the-Middle Model Training & GGUF Export Pipeline"
echo "======================================================================"
echo ""

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "❌ Error: Python is required but not found" >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
echo ""

# Check Python version
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Step 1: Run the training script
echo "📝 Step 1/2: Training FIM model..."
echo ""

if ! $PYTHON_BIN train_fim_model.py; then
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 2: Verify outputs
echo "📦 Step 2/2: Verifying outputs..."
echo ""

if [ -f "fim-model-q4_0.gguf" ]; then
    GGUF_SIZE=$(du -h fim-model-q4_0.gguf | cut -f1)
    echo "✅ GGUF model created: fim-model-q4_0.gguf ($GGUF_SIZE)"
else
    echo "⚠️  Warning: GGUF file not found"
    echo "   You may need to manually convert using llama.cpp"
fi

if [ -f "Modelfile.fim" ]; then
    echo "✅ Modelfile created: Modelfile.fim"
else
    echo "⚠️  Warning: Modelfile.fim not found"
fi

if [ -d "fim-model-lora" ]; then
    echo "✅ LoRA adapter saved: fim-model-lora/"
fi

if [ -d "fim-model-merged" ]; then
    echo "✅ Merged model saved: fim-model-merged/"
fi

echo ""
echo "======================================================================"
echo "✅ FIM MODEL READY!"
echo "======================================================================"
echo ""
echo "📦 Your quantized GGUF model: fim-model-q4_0.gguf"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Install in Ollama:"
echo "   ollama create fim-code-completion -f Modelfile.fim"
echo ""
echo "2. Test the model:"
echo "   ollama run fim-code-completion"
echo ""
echo "3. Run automated tests:"
echo "   python test_fim_model.py"
echo ""
echo "4. Interactive testing:"
echo "   python test_fim_model.py --interactive"
echo ""
echo "📚 Full documentation: FIM_MODEL_GUIDE.md"
echo ""
