#!/usr/bin/env bash
set -euo pipefail

# Ensure commands run from project root
cd "$(dirname "$0")"

if command -v python &>/dev/null; then
  PYTHON_BIN="python"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
else
  echo "Error: Python is required to run this script." >&2
  exit 1
fi

printf '\n[1/4] Generating Jarvis lab training data...\n\n'
"$PYTHON_BIN" generate_lab_training_data.py

printf '\n[2/4] Fine-tuning base model with LoRA...\n\n'
"$PYTHON_BIN" finetune_ben_lab.py

printf '\n[3/4] Converting LoRA adapter to GGUF using llama.cpp...\n\n'
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi
"$PYTHON_BIN" llama.cpp/scripts/convert_lora_to_gguf.py \
  --adapter-dir ben-lab-lora \
  --outfile ben-lab-adapter.gguf

printf '\n[4/4] Building Ollama model '\''ben-lab'\''...\n\n'
cat > Modelfile <<'EOF'
FROM llama3.2:1b

ADAPTER ./ben-lab-adapter.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM """
You are Ben's Lab AI (Jarvis-2v).
You understand the Jarvis-2v quantum phase simulator, TRI, replay drift, clustering,
and the lab API. You explain results clearly and use Ben's terminology.
"""
EOF

if ! command -v ollama &>/dev/null; then
  echo "Warning: ollama CLI not found. Skipping model creation." >&2
  echo "You can run 'ollama create ben-lab -f Modelfile' manually once Ollama is installed." >&2
else
  ollama create ben-lab -f Modelfile
fi

echo '\n✅ Done. To chat with the model, run:  ollama run ben-lab\n'
