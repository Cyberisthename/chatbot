#!/bin/bash
# JARVIS Quantum LLM - Full Training and Deployment Script
# This script trains a ChatGPT-scale Quantum LLM from scratch and deploys to Hugging Face

set -e

echo "================================================================================"
echo "🌌 JARVIS QUANTUM LLM - FULL TRAINING & DEPLOYMENT PIPELINE"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. ✅ Download 160,000+ documents (Wikipedia, books, papers)"
echo "  2. ✅ Build ChatGPT-scale Quantum Transformer (100M+ params)"
echo "  3. ✅ Train with REAL backpropagation (10 epochs)"
echo "  4. ✅ Save trained model and metrics"
echo "  5. ✅ Prepare for Hugging Face deployment"
echo ""
echo "⚠️  WARNING: This will take 20-40 hours depending on your hardware!"
echo ""
read -p "Press ENTER to begin training, or Ctrl+C to cancel..."
echo ""

# Check Python
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+."
    exit 1
fi
echo "✅ Python found: $(python3 --version)"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy..."
    pip install numpy>=1.24.0
}
echo "✅ NumPy installed"
echo ""

# Run training
echo "================================================================================"
echo "🚀 PHASE 1: DATA ACQUISITION"
echo "================================================================================"
echo "Downloading massive training corpus..."
echo ""

python3 train_full_quantum_llm_production.py

echo ""
echo "================================================================================"
echo "🎉 TRAINING COMPLETE!"
echo "================================================================================"
echo ""

# Check if model was created
if [ -f "quantum_llm_production/jarvis_quantum_llm_final.npz" ]; then
    echo "✅ Model trained successfully!"
    echo "   Location: quantum_llm_production/jarvis_quantum_llm_final.npz"
    echo ""
    
    # Copy to HF directory
    echo "📦 Copying model to Hugging Face directory..."
    cp quantum_llm_production/jarvis_quantum_llm_final.npz jarvis_quantum_ai_hf_ready/jarvis_quantum_llm.npz
    cp quantum_llm_production/tokenizer.json jarvis_quantum_ai_hf_ready/tokenizer.json
    cp quantum_llm_production/config.json jarvis_quantum_ai_hf_ready/config.json
    echo "✅ Model copied to jarvis_quantum_ai_hf_ready/"
    echo ""
    
    echo "================================================================================"
    echo "🚀 READY FOR DEPLOYMENT!"
    echo "================================================================================"
    echo ""
    echo "NEXT STEPS:"
    echo ""
    echo "1. Test locally:"
    echo "   cd jarvis_quantum_ai_hf_ready"
    echo "   python app_quantum_llm.py"
    echo ""
    echo "2. Deploy to Hugging Face:"
    echo "   cd jarvis_quantum_ai_hf_ready"
    echo "   git init"
    echo "   huggingface-cli login"
    echo "   git remote add origin https://huggingface.co/YOUR_USERNAME/jarvis-quantum-llm"
    echo "   git add ."
    echo "   git commit -m 'Deploy JARVIS Quantum LLM'"
    echo "   git push origin main"
    echo ""
    echo "3. Access your deployed model:"
    echo "   https://huggingface.co/spaces/YOUR_USERNAME/jarvis-quantum-llm"
    echo ""
    echo "================================================================================"
    echo "FOR SCIENCE! 🔬"
    echo "================================================================================"
else
    echo "⚠️  Model file not found. Training may have been interrupted."
    echo "   Check quantum_llm_production/ for partial progress."
fi
