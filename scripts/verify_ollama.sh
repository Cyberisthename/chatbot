#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "❌ ollama not found in PATH"
  echo "Install it from https://ollama.com and re-run."
  exit 1
fi

echo "🔎 Checking Ollama API: http://localhost:11434"
if ! curl -fsS "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "⚠️  Ollama server not reachable. Starting 'ollama serve' in the background..."
  mkdir -p logs
  nohup ollama serve > logs/ollama-serve.log 2>&1 &
  sleep 2
fi

curl -fsS "http://localhost:11434/api/tags" | head -c 500 || true
echo ""

echo "\n🏗️  Creating model 'jarvis' from ./Modelfile"
ollama create jarvis -f Modelfile

echo "\n▶️  Running a quick prompt"
ollama run jarvis "Hello Jarvis. Introduce yourself in one paragraph." 

echo "\n✅ Ollama is reachable and jarvis runs"
