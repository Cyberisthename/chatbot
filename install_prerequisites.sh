#!/bin/bash

# ==============================================================================
# JARVIS AI - Prerequisites Installer
# This script helps install all necessary dependencies
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     JARVIS AI - Prerequisites Installer                       ║
║                                                               ║
║     Installing: Python • Node.js • Ollama                    ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        elif [ -f /etc/arch-release ]; then
            OS="arch"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="mac"
    else
        OS="unknown"
    fi
    print_info "Detected OS: $OS"
}

check_and_install_nodejs() {
    print_info "Checking Node.js..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v)
        print_success "Node.js is installed: $NODE_VERSION"
        return 0
    fi
    
    print_warning "Node.js is not installed"
    print_info "Installing Node.js..."
    
    if [[ "$OS" == "mac" ]]; then
        if command -v brew &> /dev/null; then
            brew install node
        else
            print_error "Homebrew not found. Please install from https://nodejs.org"
            return 1
        fi
    elif [[ "$OS" == "debian" ]]; then
        sudo apt update
        sudo apt install -y nodejs npm
    elif [[ "$OS" == "redhat" ]]; then
        sudo dnf install -y nodejs npm
    elif [[ "$OS" == "arch" ]]; then
        sudo pacman -S nodejs npm
    else
        print_error "Please install Node.js from https://nodejs.org"
        return 1
    fi
    
    if command -v node &> /dev/null; then
        print_success "Node.js installed successfully"
    else
        print_error "Failed to install Node.js"
        return 1
    fi
}

check_and_install_python() {
    print_info "Checking Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python is installed: $PYTHON_VERSION"
        return 0
    fi
    
    print_warning "Python 3 is not installed"
    print_info "Installing Python 3..."
    
    if [[ "$OS" == "mac" ]]; then
        if command -v brew &> /dev/null; then
            brew install python
        else
            print_error "Homebrew not found. Please install from https://python.org"
            return 1
        fi
    elif [[ "$OS" == "debian" ]]; then
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv
    elif [[ "$OS" == "redhat" ]]; then
        sudo dnf install -y python3 python3-pip
    elif [[ "$OS" == "arch" ]]; then
        sudo pacman -S python python-pip
    else
        print_error "Please install Python 3 from https://python.org"
        return 1
    fi
    
    if command -v python3 &> /dev/null; then
        print_success "Python installed successfully"
    else
        print_error "Failed to install Python"
        return 1
    fi
}

install_ollama() {
    print_info "Checking Ollama..."
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        print_success "Ollama is installed: $OLLAMA_VERSION"
        return 0
    fi
    
    echo -ne "${YELLOW}Install Ollama? (y/N): ${NC}"
    read -r INSTALL_OLLAMA
    
    if [[ ! "$INSTALL_OLLAMA" =~ ^[Yy]$ ]]; then
        print_info "Skipping Ollama installation"
        return 0
    fi
    
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
        print_info "To start Ollama service, run: ollama serve"
    else
        print_error "Failed to install Ollama"
        return 1
    fi
}

install_python_deps() {
    print_info "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        if command -v pip3 &> /dev/null; then
            pip3 install -r requirements.txt
            print_success "Python dependencies installed"
        else
            print_warning "pip3 not found, skipping Python dependencies"
        fi
    else
        print_warning "requirements.txt not found, skipping Python dependencies"
    fi
}

install_node_deps() {
    print_info "Installing Node dependencies..."
    
    if command -v npm &> /dev/null; then
        npm install
        print_success "Node dependencies installed"
    else
        print_warning "npm not found, skipping Node dependencies"
    fi
}

main() {
    print_banner
    
    # Detect OS
    detect_os
    
    # Install components
    echo ""
    print_info "Installing prerequisites..."
    echo ""
    
    check_and_install_python
    check_and_install_nodejs
    install_ollama
    
    echo ""
    print_info "Installing project dependencies..."
    echo ""
    
    install_python_deps
    install_node_deps
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✅ Installation Complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}  Next Steps:${NC}"
    echo -e "${BLUE}  1. Run: ${GREEN}./run_ai.sh${NC}"
    echo -e "${BLUE}  2. Select your backend (Ollama, Local, or Demo)${NC}"
    echo -e "${BLUE}  3. Start chatting at http://localhost:3001${NC}"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

main "$@"
