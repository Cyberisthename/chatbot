#!/bin/bash

# Jarvis Quantum LLM - Ollama Setup Script
# Automated setup for getting Jarvis running in Ollama

set -e

echo "🤖 JARVIS QUANTUM LLM - Ollama Setup"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Ollama is installed
echo -e "${CYAN}📋 Checking prerequisites...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama is not installed!${NC}"
    echo ""
    echo "Please install Ollama first:"
    echo "  Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Or visit: https://ollama.ai/download"
    exit 1
fi
echo -e "${GREEN}✅ Ollama is installed${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 is available${NC}"

# Install Python dependencies
echo ""
echo -e "${CYAN}📦 Installing Python dependencies...${NC}"
pip3 install -q numpy requests
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check if model files exist
echo ""
echo -e "${CYAN}🔍 Checking for model files...${NC}"
if [ ! -f "../ready-to-deploy-hf/jarvis_quantum_llm.npz" ]; then
    echo -e "${RED}❌ Model weights not found!${NC}"
    echo "Please train the model first (see parent directory)"
    exit 1
fi
echo -e "${GREEN}✅ Model weights found${NC}"

# Convert model to GGUF
echo ""
echo -e "${CYAN}🔄 Converting model to GGUF format...${NC}"
python3 numpy_to_gguf.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Conversion failed!${NC}"
    exit 1
fi

# Check if GGUF was created
if [ ! -f "jarvis-quantum.gguf" ]; then
    echo -e "${RED}❌ GGUF file not created!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ GGUF file created${NC}"

# Create Ollama model
echo ""
echo -e "${CYAN}🚀 Creating Ollama model...${NC}"
ollama create jarvis -f Modelfile
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create model!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Model created in Ollama${NC}"

# Run tests
echo ""
echo -e "${CYAN}🧪 Running tests...${NC}"
python3 test_ollama.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Some tests failed, but model may still work${NC}"
else
    echo -e "${GREEN}✅ All tests passed!${NC}"
fi

# Success message
echo ""
echo -e "${GREEN}======================================"
echo "🎉 Setup Complete!"
echo "======================================${NC}"
echo ""
echo "Your Jarvis Quantum LLM is ready!"
echo ""
echo "To use:"
echo -e "  ${CYAN}ollama run jarvis${NC}"
echo ""
echo "Example prompts:"
echo "  - What is quantum mechanics?"
echo "  - Explain neural networks"
echo "  - Tell me about DNA"
echo ""
echo "Have fun! 🚀"
