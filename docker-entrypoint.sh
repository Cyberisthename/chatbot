#!/bin/bash

# JARVIS-2v Docker Entrypoint
# Starts both backend and frontend services

set -e

echo "🌟 Starting JARVIS-2v in Docker..."

# Start backend
cd /app/backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "⏳ Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    sleep 1
done

# Start frontend
cd /app/frontend
npm start &
FRONTEND_PID=$!

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "🚀 JARVIS-2v is running!"
echo "📍 Backend: http://localhost:8000"
echo "📍 Frontend: http://localhost:3000"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
