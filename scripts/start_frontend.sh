#!/bin/bash

# JARVIS-2v Frontend Startup Script

set -e

echo "🎨 Starting JARVIS-2v Frontend..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/frontend"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi

echo "📦 Node version: $(node --version)"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    npm install
fi

# Set environment variables
export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:8000}"

echo "🌐 API URL: $NEXT_PUBLIC_API_URL"

# Run frontend
echo "✅ Starting Next.js dev server..."
npm run dev
