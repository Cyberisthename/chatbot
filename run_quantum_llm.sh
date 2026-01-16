#!/bin/bash
# Quick start script for Quantum LLM training and testing
# Real scientific research - no mocks, no pre-trained models

set -e  # Exit on error

echo "=========================================================================="
echo "🚀 QUANTUM LLM - FROM SCRATCH SCIENTIFIC RESEARCH"
echo "=========================================================================="
echo ""
echo "This is REAL scientific research with:"
echo "  - Neural networks built from scratch (no PyTorch/TF)"
echo "  - Quantum-inspired attention mechanisms"
echo "  - Real training on real datasets"
echo "  - Connection to JARVIS quantum engines"
echo "  - Intelligence testing with scientific logging"
echo ""
echo "=========================================================================="
echo ""

# Check Python version
python3 --version

# Install dependencies if needed
echo ""
echo "📦 Checking dependencies..."
python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy..."
    pip3 install numpy
}

# Optional: Install Hugging Face datasets for real training data
echo ""
echo "📥 Checking for Hugging Face datasets..."
python3 -c "import datasets" 2>/dev/null || {
    echo "Note: Hugging Face datasets not installed."
    echo "Will use synthetic data for training."
    echo "Install with: pip3 install datasets"
}

# Run the training pipeline
echo ""
echo "=========================================================================="
echo "Starting Quantum LLM training and testing..."
echo "=========================================================================="
echo ""

python3 train_quantum_llm.py

echo ""
echo "=========================================================================="
echo "✅ Quantum LLM research complete!"
echo "=========================================================================="
echo ""
echo "Results saved to: ./quantum_llm_logs/"
echo ""
echo "To explore results:"
echo "  - Check quantum_llm_logs/session_*/SUMMARY_REPORT.txt"
echo "  - Review quantum_llm_logs/session_*/findings.json"
echo "  - Examine quantum_llm_logs/session_*/metrics.json"
echo ""
