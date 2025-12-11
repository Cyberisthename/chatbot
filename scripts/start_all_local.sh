#!/bin/bash

# JARVIS-2v Local Development - Start Both Backend and Frontend

set -e

echo "🌟 Starting JARVIS-2v Full Stack..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill any existing processes on ports 8000 and 3000
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start backend in background
echo "🔧 Starting backend..."
bash "$SCRIPT_DIR/start_backend.sh" > /tmp/jarvis_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to be ready..."
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Backend might not be ready yet, continuing anyway..."
else
    echo "✅ Backend is ready!"
fi

# Start frontend in foreground
echo "🎨 Starting frontend..."
bash "$SCRIPT_DIR/start_frontend.sh"

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT
