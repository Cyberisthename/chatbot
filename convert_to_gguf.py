#!/usr/bin/env python3
"""Convert a trained Hugging Face model to GGUF format.

This script is a lightweight wrapper around the conversion utilities in
``train_and_export_gguf.py``. It expects that you have already trained a model
and saved it in Hugging Face format (``config.json``, ``pytorch_model.bin``,
``tokenizer.json``, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_and_export_gguf import (
    convert_to_gguf as hf_to_gguf,
    create_modelfile,
    quantize_gguf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a trained model to GGUF format")
    parser.add_argument(
        "--model-dir",
        default="jarvis-model",
        help="Directory containing the trained Hugging Face model",
    )
    parser.add_argument(
        "--output",
        help="Path to the GGUF file to create (defaults to <model-dir>.gguf)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce a quantized GGUF file (via llama-quantize)",
    )
    parser.add_argument(
        "--quant-level",
        default="Q4_0",
        help="Quantization level to use when --quantize is passed (default: Q4_0)",
    )
    parser.add_argument(
        "--quant-output",
        help="Output path for the quantized GGUF (defaults to <output>-<level>.gguf)",
    )
    parser.add_argument(
        "--create-modelfile",
        action="store_true",
        help="Create an Ollama Modelfile pointing at the generated GGUF",
    )
    parser.add_argument(
        "--model-name",
        default="jarvis-lab",
        help="Model name to use inside the generated Modelfile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)

    output_path = args.output or f"{model_dir}.gguf"
    print("🔄 Converting model to GGUF...")
    print(f"  Model directory: {model_dir}")
    print(f"  Output GGUF:     {output_path}")

    if not hf_to_gguf(str(model_dir), output_path):
        print("❌ GGUF conversion failed")
        sys.exit(1)

    final_paths = [output_path]

    if args.quantize:
        quant_output = args.quant_output or f"{Path(output_path).stem}-{args.quant_level.lower()}.gguf"
        print()
        print("🗜️  Quantizing GGUF model...")
        if quantize_gguf(output_path, quant_output, args.quant_level):
            final_paths.append(quant_output)
        else:
            print("⚠️  Quantization failed; continuing with unquantized GGUF")

    if args.create_modelfile:
        gguf_for_modelfile = final_paths[-1]
        create_modelfile(gguf_for_modelfile, args.model_name)

    print()
    print("✅ Conversion complete. Files generated:")
    for path in final_paths:
        print(f"  - {path}")
    if args.create_modelfile:
        print("  - Modelfile.jarvis-lab (Ollama configuration)")


if __name__ == "__main__":
    main()
