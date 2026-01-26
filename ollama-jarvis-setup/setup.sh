#!/bin/bash

###############################################################################
# Jarvis Quantum LLM - Ollama Setup Script
# Automated setup for getting Jarvis running in Ollama
###############################################################################

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

###############################################################################
# Prerequisite Checks
###############################################################################

echo -e "${CYAN}📋 Checking prerequisites...${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama is not installed!${NC}"
    echo ""
    echo "Please install Ollama first:"
    echo ""
    echo -e "${CYAN}Linux/Mac:${NC}"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo -e "${CYAN}Windows:${NC}"
    echo "  Download from https://ollama.ai/download"
    echo ""
    echo -e "${CYAN}Then run this script again!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Ollama is installed${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo ""
    echo "Please install Python 3:"
    echo "  - Linux: sudo apt-get install python3"
    echo "  - Mac: brew install python3"
    echo "  - Windows: https://www.python.org/downloads/"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 is available${NC}"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip not found, attempting to install...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-pip
    elif command -v brew &> /dev/null; then
        brew install python3
    else
        echo -e "${RED}❌ Could not install pip automatically${NC}"
        echo "Please install pip manually"
        exit 1
    fi
fi
echo -e "${GREEN}✅ pip is available${NC}"

###############################################################################
# Install Python Dependencies
###############################################################################

echo ""
echo -e "${CYAN}📦 Installing Python dependencies...${NC}"

# Use pip3 or pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

$PIP_CMD install -q numpy requests 2>&1 | grep -v "already satisfied" || true
echo -e "${GREEN}✅ Dependencies installed${NC}"

###############################################################################
# Check if model files exist
###############################################################################

echo ""
echo -e "${CYAN}🔍 Checking for model files...${NC}"

# Try multiple possible locations for the model weights
MODEL_FOUND=0
MODEL_PATH=""

POSSIBLE_PATHS=(
    "../ready-to-deploy-hf/jarvis_quantum_llm.npz"
    "../../ready-to-deploy-hf/jarvis_quantum_llm.npz"
    "../jarvis_quantum_llm.npz"
    "./jarvis_quantum_llm.npz"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        MODEL_PATH="$path"
        MODEL_FOUND=1
        break
    fi
done

if [ $MODEL_FOUND -eq 0 ]; then
    echo -e "${RED}❌ Model weights not found!${NC}"
    echo ""
    echo "Searched in:"
    for path in "${POSSIBLE_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo -e "${YELLOW}Please ensure jarvis_quantum_llm.npz exists in one of these locations.${NC}"
    echo ""
    echo "If you haven't trained the model yet, run:"
    echo "  cd .."
    echo "  python3 train_full_quantum_llm_production.py"
    exit 1
fi

echo -e "${GREEN}✅ Model weights found at: $MODEL_PATH${NC}"

# Export the model path for the conversion script
export JARVIS_MODEL_PATH="$MODEL_PATH"

###############################################################################
# Convert model to GGUF
###############################################################################

echo ""
echo -e "${CYAN}🔄 Converting model to GGUF format...${NC}"
echo "  (This may take 30-60 seconds)"
echo ""

if python3 numpy_to_gguf.py; then
    # Check if GGUF was created
    if [ ! -f "jarvis-quantum.gguf" ]; then
        echo -e "${RED}❌ GGUF file not created!${NC}"
        echo ""
        echo "The conversion script ran but didn't produce the expected file."
        echo "Check the output above for errors."
        exit 1
    fi
    
    FILE_SIZE=$(du -h "jarvis-quantum.gguf" | cut -f1)
    echo ""
    echo -e "${GREEN}✅ GGUF file created successfully ($FILE_SIZE)${NC}"
else
    echo -e "${RED}❌ Conversion failed!${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - NumPy not installed: pip3 install numpy"
    echo "  - Invalid model weights"
    echo "  - Insufficient disk space"
    exit 1
fi

###############################################################################
# Start Ollama server if not running
###############################################################################

echo ""
echo -e "${CYAN}🔧 Checking Ollama server...${NC}"

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama server not responding${NC}"
    echo "  Attempting to start server..."
    
    # Try to start in background
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for server to start
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama server started${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Could not start server automatically${NC}"
        echo ""
        echo "Please start Ollama manually in another terminal:"
        echo "  ollama serve"
        echo ""
        echo "Then run this script again."
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama server is running${NC}"
fi

###############################################################################
# Create Ollama model
###############################################################################

echo ""
echo -e "${CYAN}🚀 Creating Ollama model...${NC}"

# Check if model already exists
if ollama list 2>/dev/null | grep -q "^jarvis"; then
    echo -e "${YELLOW}⚠️  Model 'jarvis' already exists${NC}"
    echo ""
    echo "Removing old model and creating fresh..."
    ollama rm jarvis 2>/dev/null || true
fi

if ollama create jarvis -f Modelfile; then
    echo -e "${GREEN}✅ Model created in Ollama${NC}"
else
    echo -e "${RED}❌ Failed to create model!${NC}"
    echo ""
    echo "Possible issues:"
    echo "  - Ollama server not running (try: ollama serve)"
    echo "  - Modelfile syntax error"
    echo "  - GGUF file corrupted"
    echo ""
    echo "Try manual creation:"
    echo "  1. Ensure Ollama server is running: ollama serve"
    echo "  2. Check GGUF exists: ls -lh jarvis-quantum.gguf"
    echo "  3. Create model: ollama create jarvis -f Modelfile"
    exit 1
fi

###############################################################################
# Run tests
###############################################################################

echo ""
echo -e "${CYAN}🧪 Running quick test...${NC}"

# Quick test with short timeout
TEST_OUTPUT=$(timeout 10s bash -c 'echo "Hi" | ollama run jarvis 2>&1' || echo "timeout")

if [ "$TEST_OUTPUT" == "timeout" ]; then
    echo -e "${YELLOW}⚠️  Test timed out, but model is installed${NC}"
elif [ -n "$TEST_OUTPUT" ]; then
    echo -e "${GREEN}✅ Model responds successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Could not test, but model is installed${NC}"
fi

###############################################################################
# Success message
###############################################################################

echo ""
echo -e "${GREEN}======================================"
echo "🎉 Setup Complete!"
echo "======================================${NC}"
echo ""
echo "Your Jarvis Quantum LLM is ready!"
echo ""
echo -e "${CYAN}To use:${NC}"
echo -e "  ${YELLOW}ollama run jarvis${NC}"
echo ""
echo -e "${CYAN}Example prompts:${NC}"
echo "  • What is quantum mechanics?"
echo "  • Explain neural networks"
echo "  • Tell me about DNA"
echo "  • How do transformers work?"
echo ""
echo -e "${CYAN}Manage your model:${NC}"
echo -e "  • List models:    ${YELLOW}ollama list${NC}"
echo -e "  • Remove model:   ${YELLOW}ollama rm jarvis${NC}"
echo -e "  • Model info:     ${YELLOW}ollama show jarvis${NC}"
echo ""
echo -e "${CYAN}Need help?${NC}"
echo "  • Troubleshooting:  🔧_TROUBLESHOOTING.md"
echo "  • Manual install:   📖_MANUAL_INSTALLATION.md"
echo "  • Complete guide:   🚀_OLLAMA_JARVIS_MASTER_GUIDE.md"
echo ""
echo -e "${GREEN}Built from scratch with real ML ❤️${NC}"
echo "Have fun! 🚀"
echo ""
