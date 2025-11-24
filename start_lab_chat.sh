#!/bin/bash
# Jarvis Lab + Ollama Chat Launcher
# Starts both the lab API and chat interface

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Jarvis Lab + Ollama Launcher${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Ensure Ollama CLI exists
if ! command -v ollama >/dev/null 2>&1; then
    echo -e "${RED}❌ Ollama CLI not found${NC}"
    echo -e "${YELLOW}Install it from https://ollama.ai/download${NC}"
    echo -e "${YELLOW}After installation, run: ${BLUE}ollama serve${NC}"
    exit 1
fi

# Select Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo -e "${YELLOW}Install Python 3.9+ and try again${NC}"
    exit 1
fi

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
else
    echo -e "${RED}❌ Ollama is not running${NC}"
    echo -e "${YELLOW}Please start Ollama first:${NC}"
    echo -e "  ${BLUE}ollama serve${NC}"
    exit 1
fi

# Check if a model is available
echo -e "${YELLOW}Checking for models...${NC}"
if ollama list 2>/dev/null | grep -q "llama3"; then
    echo -e "${GREEN}✅ Found compatible model${NC}"
elif ollama list 2>/dev/null | grep -q "qwen"; then
    echo -e "${GREEN}✅ Found compatible model${NC}"
else
    echo -e "${YELLOW}⚠️  No recommended model found${NC}"
    echo -e "${YELLOW}Consider pulling a model:${NC}"
    echo -e "  ${BLUE}ollama pull llama3.1${NC}"
fi

# Check Python dependencies
echo -e "${YELLOW}Verifying setup...${NC}"
if $PYTHON_BIN verify_setup.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Setup verification passed${NC}\n"
else
    echo -e "${RED}❌ Setup verification failed${NC}"
    echo -e "${YELLOW}Run the verification script for details:${NC}"
    echo -e "  ${BLUE}$PYTHON_BIN verify_setup.py${NC}\n"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start the lab API in the background
echo -e "${GREEN}🚀 Starting Jarvis Lab API...${NC}"
$PYTHON_BIN jarvis_api.py > logs/jarvis_api.log 2>&1 &
LAB_PID=$!
echo -e "${GREEN}   Lab API PID: ${LAB_PID}${NC}"

# Wait for lab to be ready
echo -e "${YELLOW}Waiting for lab to be ready...${NC}"
for i in {1..15}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Lab API is ready${NC}\n"
        break
    fi
    sleep 1
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ Lab API failed to start${NC}"
        echo -e "${YELLOW}Check logs/jarvis_api.log for errors${NC}"
        kill $LAB_PID 2>/dev/null
        exit 1
    fi
done

# Print URLs
echo -e "${GREEN}📡 Lab API running at: ${BLUE}http://127.0.0.1:8000${NC}"
echo -e "${GREEN}📚 API docs at: ${BLUE}http://127.0.0.1:8000/docs${NC}\n"

# Start the chat interface
echo -e "${GREEN}🤖 Starting chat interface...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}\n"

# Trap Ctrl+C to cleanup
trap "echo -e '\n${YELLOW}Stopping services...${NC}'; kill $LAB_PID 2>/dev/null; echo -e '${GREEN}✅ Services stopped${NC}'; exit 0" INT TERM

# Run chat interface (foreground)
$PYTHON_BIN chat_with_lab.py

# Cleanup if chat exits normally
kill $LAB_PID 2>/dev/null
echo -e "${GREEN}✅ Services stopped${NC}"
