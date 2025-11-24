#!/bin/bash
# Install Jarvis Lab + Ollama Dependencies
# This script installs only the required packages for the lab and chat interface

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Jarvis Lab Dependencies Installer${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}❌ Python 3.9+ is required${NC}"
    echo -e "${YELLOW}Found: Python $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}\n"

# Ask about virtual environment
echo -e "${YELLOW}Do you want to create a virtual environment? (recommended) [y/N]${NC}"
read -r USE_VENV

if [[ "$USE_VENV" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
    
    # Detect OS and provide activation instructions
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        ACTIVATE_CMD=".venv\\Scripts\\activate"
        echo -e "${GREEN}✅ Virtual environment created${NC}"
        echo -e "${YELLOW}Activate it with: ${BLUE}${ACTIVATE_CMD}${NC}\n"
        echo -e "${YELLOW}Note: On Windows, run this script in PowerShell or Git Bash${NC}\n"
    else
        ACTIVATE_CMD="source .venv/bin/activate"
        echo -e "${GREEN}✅ Virtual environment created${NC}"
        echo -e "${YELLOW}Activating...${NC}"
        source .venv/bin/activate
    fi
fi

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet

# Install core dependencies for lab + chat
echo -e "${BLUE}Installing core dependencies...${NC}"
echo -e "${YELLOW}  - fastapi (API framework)${NC}"
echo -e "${YELLOW}  - uvicorn (ASGI server)${NC}"
echo -e "${YELLOW}  - requests (HTTP client)${NC}"
echo -e "${YELLOW}  - numpy (numerical computing)${NC}"

pip install fastapi uvicorn requests numpy --quiet

echo -e "${GREEN}✅ Core dependencies installed${NC}\n"

# Ask about PyTorch
echo -e "${YELLOW}Do you want to install PyTorch? (optional, enables ML features) [y/N]${NC}"
read -r INSTALL_PYTORCH

if [[ "$INSTALL_PYTORCH" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Select PyTorch variant:${NC}"
    echo -e "  ${BLUE}1${NC}) CPU only (smaller, works everywhere)"
    echo -e "  ${BLUE}2${NC}) CUDA 11.8 (NVIDIA GPU support)"
    echo -e "  ${BLUE}3${NC}) CUDA 12.1 (latest NVIDIA GPU)"
    echo -e "  ${BLUE}4${NC}) Skip PyTorch"
    read -r PYTORCH_CHOICE
    
    case $PYTORCH_CHOICE in
        1)
            echo -e "${BLUE}Installing PyTorch (CPU)...${NC}"
            pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
            echo -e "${GREEN}✅ PyTorch (CPU) installed${NC}\n"
            ;;
        2)
            echo -e "${BLUE}Installing PyTorch (CUDA 11.8)...${NC}"
            pip install torch --index-url https://download.pytorch.org/whl/cu118 --quiet
            echo -e "${GREEN}✅ PyTorch (CUDA 11.8) installed${NC}\n"
            ;;
        3)
            echo -e "${BLUE}Installing PyTorch (CUDA 12.1)...${NC}"
            pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
            echo -e "${GREEN}✅ PyTorch (CUDA 12.1) installed${NC}\n"
            ;;
        *)
            echo -e "${YELLOW}⚠️  Skipping PyTorch (ML features will use fallback)${NC}\n"
            ;;
    esac
fi

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python3 verify_setup.py

echo -e "\n${BLUE}======================================${NC}"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${BLUE}======================================${NC}\n"

if [[ "$USE_VENV" =~ ^[Yy]$ ]] && [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
    echo -e "${YELLOW}Virtual environment is active.${NC}"
    echo -e "${YELLOW}To deactivate: ${BLUE}deactivate${NC}\n"
fi

echo -e "${GREEN}Next steps:${NC}"
echo -e "1. Install and start Ollama: ${BLUE}ollama serve${NC}"
echo -e "2. Pull a model: ${BLUE}ollama pull llama3.1${NC}"
echo -e "3. Start the lab: ${BLUE}./start_lab_chat.sh${NC}"
echo -e "\n📖 See ${BLUE}SETUP_JARVIS_LAB_OLLAMA.md${NC} for detailed instructions."
