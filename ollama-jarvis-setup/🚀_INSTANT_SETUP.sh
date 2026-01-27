#!/bin/bash

###############################################################################
# JARVIS QUANTUM LLM - ONE-COMMAND SETUP
# The easiest way to get Jarvis running on Ollama
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}${BOLD}"
cat << "EOF"
   ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
   ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
   ██║███████║██████╔╝██║   ██║██║███████╗
   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
   ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║
   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
   
   Quantum LLM - Instant Ollama Setup
   From-Scratch Training • Real Backpropagation
EOF
echo -e "${NC}"
echo ""

###############################################################################
# Step 1: Check Prerequisites
###############################################################################

echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}📋 Step 1/6: Checking Prerequisites${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Ollama
echo -n "  Checking for Ollama... "
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Found${NC}"
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "    ${CYAN}Version: $OLLAMA_VERSION${NC}"
else
    echo -e "${RED}❌ Not found${NC}"
    echo ""
    echo -e "${YELLOW}${BOLD}⚠️  Ollama is not installed!${NC}"
    echo ""
    echo -e "${BOLD}Install Ollama now:${NC}"
    echo ""
    echo -e "${CYAN}  Linux/Mac:${NC}"
    echo -e "    curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo -e "${CYAN}  Windows:${NC}"
    echo -e "    Download from https://ollama.ai/download"
    echo ""
    echo -e "${CYAN}  Then run this script again!${NC}"
    exit 1
fi

# Check Python
echo -n "  Checking for Python 3... "
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✅ Found${NC}"
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "    ${CYAN}Version: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Not found${NC}"
    echo ""
    echo -e "${RED}Python 3 is required but not installed!${NC}"
    exit 1
fi

# Check pip
echo -n "  Checking for pip... "
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✅ Found${NC}"
else
    echo -e "${YELLOW}⚠️  Not found (trying pip)${NC}"
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}❌ Neither pip3 nor pip found!${NC}"
        exit 1
    fi
    alias pip3=pip
fi

echo ""
echo -e "${GREEN}✅ All prerequisites met!${NC}"
sleep 1

###############################################################################
# Step 2: Install Python Dependencies
###############################################################################

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}📦 Step 2/6: Installing Python Dependencies${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "  Installing numpy and requests..."
pip3 install -q numpy requests 2>&1 | grep -v "already satisfied" || true
echo -e "${GREEN}  ✅ Dependencies installed${NC}"
sleep 1

###############################################################################
# Step 3: Verify Model Files
###############################################################################

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}🔍 Step 3/6: Verifying Model Files${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check for model weights
echo -n "  Looking for trained weights... "
if [ -f "../ready-to-deploy-hf/jarvis_quantum_llm.npz" ]; then
    SIZE=$(du -h "../ready-to-deploy-hf/jarvis_quantum_llm.npz" | cut -f1)
    echo -e "${GREEN}✅ Found ($SIZE)${NC}"
else
    echo -e "${RED}❌ Not found${NC}"
    echo ""
    echo -e "${RED}${BOLD}Error: Model weights not found!${NC}"
    echo ""
    echo "Expected location: ../ready-to-deploy-hf/jarvis_quantum_llm.npz"
    echo ""
    echo "You need to train the model first. See the parent directory"
    echo "for training scripts."
    exit 1
fi

# Check for config
echo -n "  Looking for model config... "
if [ -f "../ready-to-deploy-hf/config.json" ]; then
    echo -e "${GREEN}✅ Found${NC}"
else
    echo -e "${YELLOW}⚠️  Not found (will use defaults)${NC}"
fi

echo ""
echo -e "${GREEN}✅ All files present!${NC}"
sleep 1

###############################################################################
# Step 4: Convert to GGUF Format
###############################################################################

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}🔄 Step 4/6: Converting Model to GGUF Format${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "  Running conversion script..."
echo "  (This may take 30-60 seconds)"
echo ""

if python3 numpy_to_gguf.py; then
    echo ""
    if [ -f "jarvis-quantum.gguf" ]; then
        SIZE=$(du -h "jarvis-quantum.gguf" | cut -f1)
        echo -e "${GREEN}  ✅ Conversion successful! ($SIZE)${NC}"
    else
        echo -e "${RED}  ❌ GGUF file not created!${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ❌ Conversion failed!${NC}"
    echo ""
    echo "See error messages above for details."
    exit 1
fi

sleep 1

###############################################################################
# Step 5: Create Ollama Model
###############################################################################

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}🚀 Step 5/6: Creating Ollama Model${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if model already exists
if ollama list | grep -q "^jarvis"; then
    echo -e "  ${YELLOW}⚠️  Model 'jarvis' already exists${NC}"
    echo ""
    echo -n "  Would you like to recreate it? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "  Removing old model..."
        ollama rm jarvis
        echo -e "  ${GREEN}✅ Old model removed${NC}"
    else
        echo "  Skipping model creation (using existing)"
        SKIP_CREATE=1
    fi
fi

if [ -z "$SKIP_CREATE" ]; then
    echo "  Creating model from Modelfile..."
    echo "  (This may take 10-20 seconds)"
    echo ""
    
    if ollama create jarvis -f Modelfile; then
        echo ""
        echo -e "${GREEN}  ✅ Model created successfully!${NC}"
    else
        echo ""
        echo -e "${RED}  ❌ Failed to create model!${NC}"
        echo ""
        echo "Possible issues:"
        echo "  - Ollama server not running (try: ollama serve)"
        echo "  - Modelfile syntax error"
        echo "  - GGUF file corrupted"
        exit 1
    fi
fi

sleep 1

###############################################################################
# Step 6: Verify Installation
###############################################################################

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}🧪 Step 6/6: Verifying Installation${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if model is in list
echo -n "  Checking model registration... "
if ollama list | grep -q "^jarvis"; then
    echo -e "${GREEN}✅ Registered${NC}"
else
    echo -e "${RED}❌ Not found in ollama list${NC}"
    exit 1
fi

# Optional: Run quick test
echo ""
echo -e "  ${CYAN}Running quick test...${NC}"
echo ""
echo "  Prompt: 'What is 2+2?'"
echo -e "  ${YELLOW}Response:${NC}"
echo ""
TEST_OUTPUT=$(echo "What is 2+2?" | ollama run jarvis --verbose 2>/dev/null | head -5 || echo "Test skipped")
echo "$TEST_OUTPUT" | sed 's/^/    /'
echo ""

if [ "$TEST_OUTPUT" != "Test skipped" ]; then
    echo -e "${GREEN}  ✅ Model responds successfully!${NC}"
else
    echo -e "${YELLOW}  ⚠️  Could not test (but model is installed)${NC}"
fi

###############################################################################
# SUCCESS!
###############################################################################

echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}           🎉 SETUP COMPLETE! 🎉${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BOLD}Your Jarvis Quantum LLM is ready to use!${NC}"
echo ""
echo -e "${CYAN}${BOLD}Start chatting:${NC}"
echo -e "  ${YELLOW}ollama run jarvis${NC}"
echo ""
echo -e "${CYAN}${BOLD}Example prompts:${NC}"
echo "  • What is quantum mechanics?"
echo "  • Explain neural networks"
echo "  • How does DNA work?"
echo "  • Tell me about black holes"
echo ""
echo -e "${CYAN}${BOLD}Manage your model:${NC}"
echo "  • List models:    ${YELLOW}ollama list${NC}"
echo "  • Remove model:   ${YELLOW}ollama rm jarvis${NC}"
echo "  • Model info:     ${YELLOW}ollama show jarvis${NC}"
echo ""
echo -e "${CYAN}${BOLD}Need help?${NC}"
echo "  • Manual setup:      ${YELLOW}📖_MANUAL_INSTALLATION.md${NC}"
echo "  • Troubleshooting:   ${YELLOW}🔧_TROUBLESHOOTING.md${NC}"
echo "  • Full guide:        ${YELLOW}🚀_OLLAMA_JARVIS_MASTER_GUIDE.md${NC}"
echo ""
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  Built from scratch with real machine learning ❤️${NC}"
echo -e "${MAGENTA}  Every parameter trained through backpropagation${NC}"
echo -e "${MAGENTA}  No pre-trained weights • 100% transparent${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Happy chatting! 🚀${NC}"
echo ""
