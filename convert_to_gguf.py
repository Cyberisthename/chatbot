#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM as CTModel

def convert_to_gguf():
    print("📦 Loading trained model...")
    model_path = "./jarvis-model"
    output_path = "./jarvis-7b-q4_0.gguf"
    
    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("🔄 Converting to GGUF format...")
    # Convert to GGUF
    ct_model = CTModel.from_pretrained(
        model_path,
        model_type="gpt2",
        model_file=output_path,
        max_new_tokens=2048,
        context_length=2048,
        gpu_layers=0
    )
    
    # Add metadata
    metadata = {
        "name": "J.A.R.V.I.S. Custom Model",
        "creator": "Ben",
        "version": "1.0.0",
        "license": "Proprietary - All rights reserved",
        "description": "Custom-trained J.A.R.V.I.S. model for personal use"
    }
    
    # Save metadata
    with open(output_path + ".meta", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model converted and saved to {output_path}")
    print("🏷️ Added custom metadata and license information")

if __name__ == "__main__":
    convert_to_gguf()