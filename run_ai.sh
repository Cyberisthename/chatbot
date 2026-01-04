#!/bin/bash

# ==============================================================================
# JARVIS AI System - Easy Run Script
# Supports: Ollama, Pinokio, and Local Inference
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗  ██╗ ║
║     ██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║ ║
║     █████╗  ██║   ██║███████╗██████╔╝███████║   ██║   ███████║ ║
║     ██╔══╝  ██║   ██║╚════██║██╔══██╗██╔══██║   ██║   ██╔══██║ ║
║     ██║     ╚██████╔╝███████║██████╔╝██║  ██║   ██║   ██║  ██║ ║
║     ╚═╝      ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ║
║                                                               ║
║                    Easy Run Script v1.0                       ║
║              Ollama • Pinokio • Local Inference               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Print section header
print_header() {
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print info message
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect available AI backends
detect_backends() {
    AVAILABLE_BACKENDS=()
    
    if command_exists ollama; then
        AVAILABLE_BACKENDS+=("ollama")
        print_success "Ollama detected"
    fi
    
    if command_exists pinokio; then
        AVAILABLE_BACKENDS+=("pinokio")
        print_success "Pinokio detected"
    fi
    
    # Always add local as an option
    AVAILABLE_BACKENDS+=("local")
    print_info "Local inference available"
    
    # Check for Python
    if command_exists python3; then
        print_success "Python 3 detected"
    else
        print_error "Python 3 not found - Local inference requires Python"
    fi
    
    # Check for Node.js
    if command_exists node; then
        print_success "Node.js detected"
    else
        print_error "Node.js not found - Web UI requires Node.js"
    fi
}

# Show menu
show_menu() {
    echo -e "\n${GREEN}Available AI Backends:${NC}"
    for i in "${!AVAILABLE_BACKENDS[@]}"; do
        backend="${AVAILABLE_BACKENDS[$i]}"
        case $backend in
            ollama)
                echo -e "  ${GREEN}$((i+1)).${NC} Ollama (Recommended for easy setup)"
                ;;
            pinokio)
                echo -e "  ${GREEN}$((i+1)).${NC} Pinokio"
                ;;
            local)
                echo -e "  ${GREEN}$((i+1)).${NC} Local Inference (Python backend)"
                ;;
        esac
    done
    echo -e "  ${YELLOW}Q.${NC} Quit"
    echo -ne "\n${BLUE}Select a backend (1-${#AVAILABLE_BACKENDS[@]} or Q): ${NC}"
    read -r choice
}

# Setup Ollama
setup_ollama() {
    print_header "Setting up Ollama"
    
    # Check if Ollama is installed
    if ! command_exists ollama; then
        print_warning "Ollama is not installed"
        print_info "To install Ollama, visit: https://ollama.ai/download"
        print_info "Or run: curl -fsSL https://ollama.ai/install.sh | sh"
        exit 1
    fi
    
    # Check if ollama service is running
    if ! ollama list >/dev/null 2>&1; then
        print_warning "Ollama service is not running"
        print_info "Starting Ollama service..."
        ollama serve &
        sleep 3
    fi
    
    # List available models
    print_info "Available Ollama models:"
    ollama list
    
    # Ask for model or use default
    echo -ne "\n${BLUE}Enter model name (default: llama3.2): ${NC}"
    read -r MODEL_NAME
    MODEL_NAME=${MODEL_NAME:-"llama3.2"}
    
    # Pull model if not available
    if ! ollama list | grep -q "$MODEL_NAME"; then
        print_info "Pulling model $MODEL_NAME..."
        ollama pull "$MODEL_NAME"
    fi
    
    OLLAMA_MODEL="$MODEL_NAME"
    print_success "Using Ollama with model: $OLLAMA_MODEL"
}

# Setup Pinokio
setup_pinokio() {
    print_header "Setting up Pinokio"
    
    if ! command_exists pinokio; then
        print_warning "Pinokio is not installed"
        print_info "To install Pinokio, visit: https://pinokio.computer"
        exit 1
    fi
    
    print_info "Starting Pinokio..."
    pinokio &
    PINOKIO_PID=$!
    print_success "Pinokio started (PID: $PINOKIO_PID)"
    
    sleep 2
    print_info "Pinokio should now be accessible in your browser"
}

# Setup Local Inference
setup_local() {
    print_header "Setting up Local Inference"
    
    # Check for requirements
    if ! command_exists python3; then
        print_error "Python 3 is required for local inference"
        exit 1
    fi
    
    # Install Python dependencies
    print_info "Checking Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -q -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
    
    # Check for GGUF model
    print_info "Checking for GGUF models..."
    if [ -f "models/jarvis-7b-q4_0.gguf" ]; then
        MODEL_PATH="models/jarvis-7b-q4_0.gguf"
        print_success "Found model: $MODEL_PATH"
    elif [ -f "jarvis-model/jarvis-7b-q4_0.gguf" ]; then
        MODEL_PATH="jarvis-model/jarvis-7b-q4_0.gguf"
        print_success "Found model: $MODEL_PATH"
    else
        print_warning "No GGUF model found"
        print_info "You can:"
        print_info "  1. Download a GGUF model from HuggingFace"
        print_info "  2. Train and export your own model using train_and_export_gguf.py"
        print_info "  3. Convert a HuggingFace model using convert_to_gguf.py"
        
        echo -ne "\n${BLUE}Continue without model (mock mode)? [y/N]: ${NC}"
        read -r CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            exit 1
        fi
        MODEL_PATH=""
    fi
    
    # Update config if model path found
    if [ -n "$MODEL_PATH" ]; then
        print_info "Updating config.yaml with model path..."
        if command_exists python3; then
            python3 -c "
import yaml
import os
config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['path'] = '$MODEL_PATH'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print('Config updated successfully')
" 2>/dev/null || true
        fi
    fi
    
    # Start Python backend
    print_info "Starting Python inference backend..."
    if [ -f "inference.py" ]; then
        python3 inference.py > logs/inference.log 2>&1 &
        INFERENCE_PID=$!
        print_success "Inference backend started (PID: $INFERENCE_PID)"
    else
        print_warning "inference.py not found - backend will run in mock mode"
    fi
}

# Start Web UI
start_web_ui() {
    print_header "Starting Web UI"
    
    # Create logs directory
    mkdir -p logs
    
    # Check for Node.js
    if ! command_exists node; then
        print_error "Node.js is required for the web UI"
        return 1
    fi
    
    # Install Node dependencies if needed
    if [ ! -d "node_modules" ]; then
        print_info "Installing Node dependencies..."
        npm install --silent
        print_success "Node dependencies installed"
    fi
    
    # Start the server
    print_info "Starting web server on http://localhost:3001..."
    node server.js > logs/server.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 3
    
    if ps -p $SERVER_PID > /dev/null; then
        print_success "Web server started (PID: $SERVER_PID)"
        echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}  🚀 JARVIS AI is now running!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}  📱 Web Interface:  http://localhost:3001${NC}"
        echo -e "${BLUE}  📊 API Endpoint:   http://localhost:3001/api/chat${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"
        
        # Save PIDs for cleanup
        echo "$SERVER_PID" > logs/server.pid
        [ -n "$INFERENCE_PID" ] && echo "$INFERENCE_PID" >> logs/server.pid
        [ -n "$PINOKIO_PID" ] && echo "$PINOKIO_PID" >> logs/server.pid
        
        return 0
    else
        print_error "Failed to start web server"
        cat logs/server.log
        return 1
    fi
}

# Cleanup on exit
cleanup() {
    print_header "Stopping Services"
    
    if [ -f "logs/server.pid" ]; then
        while read -r pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                print_info "Stopping process $pid..."
                kill "$pid" 2>/dev/null || true
            fi
        done < logs/server.pid
        rm logs/server.pid
    fi
    
    print_success "All services stopped"
    exit 0
}

# Main execution
main() {
    print_banner
    
    # Set up trap for cleanup
    trap cleanup SIGINT SIGTERM
    
    # Detect available backends
    print_header "Detecting AI Backends"
    detect_backends
    
    # Show menu
    if [ "${#AVAILABLE_BACKENDS[@]}" -gt 1 ]; then
        show_menu
        case $choice in
            [Qq])
                print_info "Exiting..."
                exit 0
                ;;
            [1-${#AVAILABLE_BACKENDS[@]}])
                BACKEND="${AVAILABLE_BACKENDS[$((choice-1))]}"
                ;;
            *)
                print_error "Invalid choice"
                exit 1
                ;;
        esac
    else
        BACKEND="${AVAILABLE_BACKENDS[0]}"
        print_info "Auto-selected backend: $BACKEND"
    fi
    
    # Setup selected backend
    case $BACKEND in
        ollama)
            setup_ollama
            export AI_BACKEND="ollama"
            export AI_MODEL="$OLLAMA_MODEL"
            ;;
        pinokio)
            setup_pinokio
            export AI_BACKEND="pinokio"
            ;;
        local)
            setup_local
            export AI_BACKEND="local"
            ;;
    esac
    
    # Start Web UI
    start_web_ui
    
    # Wait indefinitely
    wait
}

# Run main function
main "$@"
