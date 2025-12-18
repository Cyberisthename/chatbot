"""train_jarvis.py

Legacy entrypoint for training the J.A.R.V.I.S. model.

This repository originally trained a small GPT-2 variant. To produce an Ollama/
llama.cpp compatible GGUF, training must use a llama.cpp-convertible base model
(Llama/Mistral/Qwen2/Gemma/Phi) and fine-tune via LoRA/QLoRA.

This script is now a thin wrapper around `train_llm.py` to keep the existing
project structure intact.

Run:
  python3 train_jarvis.py

(You can pass any `train_llm.py` CLI flags here as well.)
"""

from __future__ import annotations

import torch

from train_llm import main as train_llm_main


def _print_system_info() -> None:
    print("🤖 J.A.R.V.I.S. Training System")
    print("==============================")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Available: Yes ({props.name})")
        print(f"GPU Memory: {props.total_memory / 1e9:.1f}GB")
    else:
        print("GPU Available: No (CPU-only)")


if __name__ == "__main__":
    _print_system_info()
    train_llm_main()
