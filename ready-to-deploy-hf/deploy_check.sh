#!/bin/bash
echo "🔍 Checking JARVIS HF Deployment Package..."

# 1. Check required files
FILES=("app.py" "jarvis_quantum_llm.npz" "tokenizer.json" "config.json" "requirements.txt" "src/__init__.py")

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ] || [ -d "$FILE" ]; then
        echo "✅ Found $FILE"
    else
        echo "❌ Missing $FILE"
        exit 1
    fi
done

# 2. Check Python imports
echo "🐍 Testing Python imports..."
python3 -c "import gradio; import numpy; from src.quantum_llm import QuantumTransformer, SimpleTokenizer; print('✅ Core modules loaded successfully')"

# 3. Check model loading
echo "📥 Testing model load..."
python3 -c "from src.quantum_llm import QuantumTransformer; model = QuantumTransformer.load('jarvis_quantum_llm.npz'); print('✅ Model weights loaded successfully')"

echo "✨ JARVIS Package is VALID and READY for deployment to Hugging Face."
