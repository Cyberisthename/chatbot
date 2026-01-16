#!/bin/bash
# JARVIS Historical Knowledge Training - Quick Start
# Executes full multi-epoch training pipeline

set -e

echo "========================================="
echo "🚀 JARVIS TRAINING INITIALIZATION"
echo "========================================="
echo ""

# Check Python version
echo "🐍 Checking Python..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q datasets huggingface_hub

echo ""
echo "========================================="
echo "🧠 STARTING JARVIS TRAINING"
echo "========================================="
echo ""
echo "Configuration:"
echo "  • Dataset: institutional/institutional-books-1.0"
echo "  • Target: 50 GB (physics, medicine, quantum, 1800-1950)"
echo "  • Epochs: 3"
echo "  • Output: ./jarvis_historical_knowledge"
echo ""

# Run training pipeline
python3 jarvis_historical_training_pipeline.py \
    --output-dir ./jarvis_historical_knowledge \
    --target-size-gb 50 \
    --epochs 3

echo ""
echo "========================================="
echo "✅ TRAINING COMPLETE"
echo "========================================="
echo ""
echo "📁 Check results in: ./jarvis_historical_knowledge/"
echo "🧠 Jarvis now has infinite historical recall!"
