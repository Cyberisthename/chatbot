#!/usr/bin/env python3
"""
Demo: Ben Lab LoRA Fine-tuning Pipeline

This demonstrates the complete workflow:
1. Generate training data from Jarvis Lab API
2. Fine-tune a small LLM with LoRA
3. Test the resulting model

NOTE: This is a demo script showing how the pieces fit together.
For actual training, use the individual scripts or train_and_install.sh
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import requests


def check_jarvis_api_health() -> bool:
    """Check if Jarvis Lab API is running."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def demonstrate_data_generation() -> None:
    """Show how training data is generated."""
    print("=" * 60)
    print("STEP 1: Generate Training Data from Quantum Experiments")
    print("=" * 60)
    print()

    if not check_jarvis_api_health():
        print("❌ Jarvis Lab API not running at http://127.0.0.1:8000")
        print("Start it with: python jarvis_api.py")
        print()
        print("Skipping data generation demo...")
        return

    print("✅ Jarvis Lab API is running")
    print()

    print("Example: Running a phase experiment...")
    payload = {
        "phase_type": "ising_symmetry_breaking",
        "system_size": 32,
        "depth": 8,
        "seed": 42,
        "bias": 0.7,
    }

    response = requests.post(
        "http://127.0.0.1:8000/run_phase_experiment",
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Experiment completed: {result['experiment_id']}")
        print(f"  Phase: {result['phase_type']}")
        print(f"  Feature vector (first 5): {result['feature_vector'][:5]}")
        print()

        print("This would be converted to a training sample:")
        sample = {
            "instruction": "Explain an Ising symmetry-breaking experiment with depth=8 and bias=0.7",
            "output": f"Analysis:\n- Phase: {result['phase_type']}\n- Features: {result['feature_vector'][:5]}...\n- Interpretation: [model would learn patterns from thousands of these]"
        }
        print(json.dumps(sample, indent=2))
    else:
        print(f"❌ Experiment failed: {response.status_code}")

    print()
    print("In practice, generate_lab_training_data.py creates ~300+ samples")
    print("covering all experiment types: phase, TRI, discovery, replay drift")
    print()


def demonstrate_lora_concept() -> None:
    """Explain LoRA fine-tuning concept."""
    print("=" * 60)
    print("STEP 2: Fine-tune with LoRA (Low-Rank Adaptation)")
    print("=" * 60)
    print()

    print("LoRA is an efficient fine-tuning technique:")
    print()
    print("Traditional Fine-tuning:")
    print("  ❌ Update ALL model parameters (~1B+ parameters)")
    print("  ❌ Requires large GPU memory")
    print("  ❌ Slow training")
    print()
    print("LoRA Fine-tuning:")
    print("  ✅ Add small adapter layers (~1-5M parameters)")
    print("  ✅ Train only the adapter (freeze base model)")
    print("  ✅ Fast, memory-efficient")
    print("  ✅ Can be applied to any base model")
    print()

    print("What finetune_ben_lab.py does:")
    print("  1. Load base model (e.g., Llama-3.2-1B)")
    print("  2. Add LoRA adapter layers")
    print("  3. Train adapter on quantum experiment data")
    print("  4. Save adapter weights to ben-lab-lora/")
    print()

    print("Result: A specialized adapter that teaches the model about")
    print("quantum phases, TRI, clustering, and Ben Lab terminology.")
    print()


def demonstrate_ollama_integration() -> None:
    """Show how the adapter gets integrated into Ollama."""
    print("=" * 60)
    print("STEP 3: Convert and Install to Ollama")
    print("=" * 60)
    print()

    print("Conversion pipeline:")
    print()
    print("  [LoRA Adapter]  ->  [GGUF Format]  ->  [Ollama Model]")
    print("   HuggingFace         llama.cpp         Local runtime")
    print()

    print("1. Convert adapter to GGUF (quantized format):")
    print("   python llama.cpp/scripts/convert_lora_to_gguf.py \\")
    print("     --adapter-dir ben-lab-lora \\")
    print("     --outfile ben-lab-adapter.gguf")
    print()

    print("2. Create Ollama Modelfile:")
    print("   ```")
    print("   FROM llama3.2:1b")
    print("   ADAPTER ./ben-lab-adapter.gguf")
    print("   PARAMETER temperature 0.2")
    print("   SYSTEM \"You are Ben's Lab AI...\"")
    print("   ```")
    print()

    print("3. Build model:")
    print("   ollama create ben-lab -f Modelfile")
    print()

    print("Result: 'ben-lab' model available in Ollama!")
    print()


def demonstrate_usage() -> None:
    """Show how to use the fine-tuned model."""
    print("=" * 60)
    print("USAGE: Talking to Your Quantum-Trained Model")
    print("=" * 60)
    print()

    print("Command line:")
    print("  ollama run ben-lab")
    print()

    print("Python (requests):")
    print("  import requests")
    print("  resp = requests.post('http://localhost:11434/api/generate',")
    print("      json={'model': 'ben-lab', 'prompt': 'What is TRI?'})")
    print()

    print("Python (ollama package):")
    print("  import ollama")
    print("  response = ollama.generate(model='ben-lab', prompt='Explain SPT phases')")
    print("  print(response['response'])")
    print()

    print("Integration with chat_with_lab.py:")
    print("  DEFAULT_MODEL = 'ben-lab'  # Use your fine-tuned model")
    print()


def show_sample_prompts() -> None:
    """Show example prompts for the fine-tuned model."""
    print("=" * 60)
    print("SAMPLE PROMPTS for ben-lab Model")
    print("=" * 60)
    print()

    prompts = [
        "What is TRI and how does it measure quantum phase properties?",
        "Explain the difference between Ising symmetry-breaking and SPT cluster phases",
        "How do I interpret high vs low replay drift values?",
        "Design an experiment to maximize TRI for detecting broken symmetry",
        "What do the 16 dimensions of the feature vector represent?",
        "How does unsupervised clustering reveal phase boundaries?",
        "Compare the behavior of trivial product vs pseudorandom phases",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    print()

    print("The model learned these patterns from ~300+ quantum experiments!")
    print()


def main() -> None:
    print()
    print("╔" + "=" * 58 + "╗")
    print("║  BEN LAB: Fine-tuned LLM from Quantum Experiments Demo  ║")
    print("╚" + "=" * 58 + "╝")
    print()

    demonstrate_data_generation()
    input("Press Enter to continue...")
    print()

    demonstrate_lora_concept()
    input("Press Enter to continue...")
    print()

    demonstrate_ollama_integration()
    input("Press Enter to continue...")
    print()

    demonstrate_usage()
    input("Press Enter to continue...")
    print()

    show_sample_prompts()

    print("=" * 60)
    print("GETTING STARTED")
    print("=" * 60)
    print()
    print("Quick start (requires API running):")
    print("  ./train_and_install.sh")
    print()
    print("Step by step:")
    print("  1. python jarvis_api.py                      # Terminal 1")
    print("  2. python generate_lab_training_data.py     # Terminal 2")
    print("  3. python finetune_ben_lab.py               # Wait ~30 min")
    print("  4. Convert and install (see docs)")
    print()
    print("Documentation:")
    print("  - BEN_LAB_LORA_OLLAMA.md       (Full guide)")
    print("  - QUICK_START_BEN_LAB_LORA.md  (TL;DR)")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
