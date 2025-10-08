#!/bin/bash

# J.A.R.V.I.S. AI System Start Script

echo "🚀 Starting J.A.R.V.I.S. AI System..."

# Check if installation is complete
if [ ! -f "node_modules/express/package.json" ]; then
    echo "❌ J.A.R.V.I.S. is not installed. Please run './install.sh' first."
    exit 1
fi

# Check if server.js exists
if [ ! -f "server.js" ]; then
    echo "❌ server.js not found. Please ensure you have the complete J.A.R.V.I.S. package."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🤖 Starting J.A.R.V.I.S. AI System..."
echo "📝 Logs will be saved to logs/jarvis.log"
echo ""

# Start the Node.js server in the background
echo "🟢 Starting Node.js server (API & Web UI) on port 3001..."
npm start > logs/jarvis-node.log 2>&1 &
NODE_PID=$!

# Start the Streamlit app in the background
echo "🟣 Starting Streamlit app on port 8501..."
streamlit run streamlit_app.py > logs/jarvis-streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo ""
echo "✅ All services started!"
echo "- Node.js API/Web: http://localhost:3001"
echo "- Streamlit Chatbot: http://localhost:8501"
echo ""
echo "To stop all services, run: kill $NODE_PID $STREAMLIT_PID"