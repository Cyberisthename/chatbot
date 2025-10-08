#!/bin/bash

# J.A.R.V.I.S. AI System Installation Script

echo "🚀 Installing J.A.R.V.I.S. AI System..."

# Check system requirements
echo "📋 Checking system requirements..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    echo "📥 Download from: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js version 16+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the J.A.R.V.I.S. AI System directory"
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

# Set up permissions
echo "🔐 Setting up permissions..."
chmod +x server.js

# Create logs directory
mkdir -p logs

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 To start J.A.R.V.I.S.:"
echo "   npm start"
echo ""
echo "🌐 Web interface will be available at:"
echo "   http://localhost:3001"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "🤖 J.A.R.V.I.S. is ready to assist you!"