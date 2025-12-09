#!/bin/bash
# JARVIS-2v Jetson Orin Deployment Script
# Optimized for NVIDIA Jetson Orin NX/AGX

set -e

echo "🚀 Starting JARVIS-2v on Jetson Orin..."

# Configuration for Jetson
export JETSON_MODE="1"
export CUDA_VISIBLE_DEVICES=0

# Set Jetson-specific environment variables
export TORCH_CUDA_ARCH_LIST="7.2"  # Orin architecture
export CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1

# Set JARVIS-specific config
export USE_JETSON="1"
export JARVIS_CONFIG="./config_jetson.yaml"

# Memory management for Jetson (8GB RAM)
export JARVIS_MEMORY_LIMIT="6G"
export JARVIS_GPU_MEMORY_FRACTION="0.8"

# Jetson GPU layers (adjust based on your model)
export JARVIS_GPU_LAYERS="20"  # Start conservative, tune up to 30-40 for Orin

# Low power mode options
if [ "$1" = "--low-power" ]; then
    echo "⚡ Running in low-power mode"
    export JARVIS_MODE="low_power"
    export JARVIS_GPU_LAYERS="10"
fi

# Offline mode (no network)
if [ "$1" = "--offline" ]; then
    echo "🔒 Running in offline mode"
    export JARVIS_OFFLINE="1"
fi

# Check for llama.cpp with CUDA
if ! python3 -c "import llama_cpp_cuda" 2>/dev/null; then
    echo "⚠️  llama_cpp_cuda not found, installing..."
    pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
fi

# Run health check
python3 -c "
import torch
torch.cuda.init()
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'🔥 GPU: {torch.cuda.get_device_name(0)}')
print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Start JARVIS-2v API
echo ""
echo "🏃 Starting JARVIS-2v API server..."
echo "📊 Model: JARVIS-7B-Q4_0 (Jetson-optimized)"
echo "🔧 GPU Layers: $JARVIS_GPU_LAYERS"
echo "⚡ Mode: ${JARVIS_MODE:-standard}"
echo ""

# Run with custom config
python3 -m src.api.main --config ./config_jetson.yaml

echo "✅ JARVIS-2v Jetson deployment complete!"