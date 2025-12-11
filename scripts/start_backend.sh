#!/bin/bash

# JARVIS-2v Backend Startup Script

set -e

echo "🚀 Starting JARVIS-2v Backend..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "⚠️  config.yaml not found, backend will use defaults"
fi

# Set environment variables
export JARVIS_CONFIG="${JARVIS_CONFIG:-./config.yaml}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "📍 Project root: $PROJECT_ROOT"
echo "⚙️  Config: $JARVIS_CONFIG"
echo "🌐 Host: $HOST:$PORT"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"

# Install dependencies if needed
if [ ! -d ".venv" ] && [ ! -f "backend/requirements_installed.txt" ]; then
    echo "📦 Installing backend dependencies..."
    python3 -m pip install --break-system-packages -q -r backend/requirements.txt 2>&1 || true
    touch backend/requirements_installed.txt
fi

# Run backend
echo "✅ Starting FastAPI server..."
cd backend
python3 -m uvicorn main:app --host "$HOST" --port "$PORT" --reload
