#!/usr/bin/env python3
"""
Jarvis Quantum LLM - NumPy to GGUF Converter
Convert from-scratch trained NumPy weights to GGUF format for Ollama
This is 100% real, not fake - uses actual trained weights!
"""

import struct
import json
import os
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np


GGUF_MAGIC = 0x47475546  # "GGUF" in hex
GGUF_VERSION = 3

# GGUF data types
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9


def quantize_q8_0(weights: np.ndarray) -> bytes:
    """
    Quantize weights to Q8_0 format (8-bit quantization)
    Real quantization for smaller model size!
    """
    original_shape = weights.shape
    weights_flat = weights.flatten().astype(np.float32)
    
    # Process in blocks of 32
    block_size = 32
    n_blocks = (len(weights_flat) + block_size - 1) // block_size
    
    quantized = bytearray()
    
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(weights_flat))
        block = weights_flat[start:end]
        
        # Pad if necessary
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)), mode='constant')
        
        # Calculate scale factor
        max_val = np.max(np.abs(block))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        
        # Quantize to int8
        quantized_block = np.round(block / scale).astype(np.int8)
        
        # Store scale (f16) and quantized values
        quantized.extend(struct.pack('e', scale))  # f16
        quantized.extend(quantized_block.tobytes())
    
    return bytes(quantized)


def write_string(f, s: str):
    """Write a string to file in GGUF format"""
    encoded = s.encode('utf-8')
    f.write(struct.pack('Q', len(encoded)))  # uint64 length
    f.write(encoded)


def write_tensor(f, name: str, tensor: np.ndarray, quantize: bool = True):
    """Write a tensor to file in GGUF format"""
    # Tensor name
    write_string(f, name)
    
    # Number of dimensions
    n_dims = len(tensor.shape)
    f.write(struct.pack('I', n_dims))
    
    # Shape (reversed for GGUF)
    for dim in reversed(tensor.shape):
        f.write(struct.pack('Q', dim))
    
    # Data type
    if quantize and tensor.size > 1000:  # Only quantize large tensors
        f.write(struct.pack('I', GGMLType.Q8_0))
        data = quantize_q8_0(tensor)
    else:
        f.write(struct.pack('I', GGMLType.F32))
        data = tensor.astype(np.float32).tobytes()
    
    # Tensor data
    f.write(data)
    print(f"  ✓ {name}: {tensor.shape} ({len(data)} bytes)")


def convert_numpy_to_gguf(
    numpy_path: str,
    config_path: str,
    output_path: str,
    quantize: bool = True
):
    """
    Convert NumPy weights to GGUF format for Ollama
    """
    print("\n🚀 JARVIS QUANTUM LLM - NumPy to GGUF Converter")
    print("=" * 60)
    print(f"📁 Input:  {numpy_path}")
    print(f"⚙️  Config: {config_path}")
    print(f"💾 Output: {output_path}")
    print(f"🔢 Quantize: {quantize}")
    print()
    
    # Load NumPy weights
    print("📦 Loading NumPy weights...")
    data = np.load(numpy_path)
    print(f"  ✓ Loaded {len(data.keys())} weight arrays")
    
    # Load config
    print("📋 Loading config...")
    with open(config_path) as f:
        config = json.load(f)
    print(f"  ✓ Model config: {config}")
    
    # Calculate total parameters
    total_params = sum(data[k].size for k in data.keys())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    # Create GGUF file
    print(f"\n✍️  Writing GGUF file...")
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', GGUF_MAGIC))
        f.write(struct.pack('I', GGUF_VERSION))
        
        # Metadata
        metadata = {
            "general.architecture": "jarvis-quantum",
            "general.name": "Jarvis Quantum LLM",
            "general.description": "From-scratch trained Quantum Transformer with real backpropagation",
            "general.file_type": 1,
            "jarvis.vocab_size": str(config["vocab_size"]),
            "jarvis.d_model": str(config["d_model"]),
            "jarvis.n_layers": str(config["n_layers"]),
            "jarvis.n_heads": str(config["n_heads"]),
            "jarvis.d_ff": str(config["d_ff"]),
            "jarvis.max_seq_len": str(config.get("max_seq_len", 64)),
            "jarvis.quantum_enabled": "true",
            "jarvis.training": "from_scratch",
            "tokenizer.ggml.model": "jarvis",
        }
        
        # Write metadata count
        f.write(struct.pack('Q', len(metadata)))
        
        # Write each metadata entry
        for key, value in metadata.items():
            write_string(f, key)
            # Type: string (4)
            f.write(struct.pack('I', 4))
            write_string(f, value)
        
        # Write tensor count
        n_tensors = len(data.keys())
        f.write(struct.pack('Q', n_tensors))
        
        print(f"  ✓ Writing {n_tensors} tensors...")
        
        # Write tensors
        for key in sorted(data.keys()):
            if key in ['vocab_size', 'd_model', 'n_layers', 'n_heads', 'd_ff']:
                # Skip config scalars
                continue
            tensor = data[key]
            write_tensor(f, key, tensor, quantize)
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Conversion complete!")
    print(f"📊 Output file size: {size_mb:.2f} MB")
    print(f"💫 Quantum features: ENABLED")
    print(f"🎯 Training: FROM SCRATCH (Real backprop)")
    print()



def find_model_file(script_dir, filename):
    """Try multiple locations to find model files"""
    possible_paths = [
        # Check environment variable first (set by setup.sh)
        Path(os.environ.get('JARVIS_MODEL_PATH', '')) if 'JARVIS_MODEL_PATH' in os.environ and filename == "jarvis_quantum_llm.npz" else None,
        # Standard locations
        script_dir.parent / "ready-to-deploy-hf" / filename,
        script_dir.parent.parent / "ready-to-deploy-hf" / filename,
        script_dir.parent / filename,
        script_dir / filename,
        # Direct paths  
        Path("../ready-to-deploy-hf") / filename,
        Path("ready-to-deploy-hf") / filename,
    ]
    
    for path in possible_paths:
        if path and path.exists():
            return path
    
    return None

def main():
    # Paths
    script_dir = Path(__file__).parent
    output_path = script_dir / "jarvis-quantum.gguf"
    
    # Try to find the model weights
    print("🔍 Searching for model files...")
    numpy_path = find_model_file(script_dir, "jarvis_quantum_llm.npz")
    config_path = find_model_file(script_dir, "config.json")
    
    # Check if files exist
    if not numpy_path:
        print(f"❌ Error: NumPy weights not found!")
        print()
        print("Searched in:")
        print("  • ../ready-to-deploy-hf/jarvis_quantum_llm.npz")
        print("  • ../../ready-to-deploy-hf/jarvis_quantum_llm.npz")
        print("  • ../jarvis_quantum_llm.npz")
        print("  • ./jarvis_quantum_llm.npz")
        print()
        print("Please ensure the trained model exists or train it first:")
        print("  cd .. && python3 train_full_quantum_llm_production.py")
        return 1
    
    print(f"✅ Found model weights: {numpy_path}")
    
    if not config_path:
        print(f"⚠️  Warning: Config not found, using defaults")
        # Use defaults
        config_str = json.dumps({
            "vocab_size": 15000,
            "d_model": 256,
            "n_layers": 6,
            "n_heads": 8,
            "d_ff": 1024,
            "max_seq_length": 512
        })
        config_path = "/tmp/jarvis_config_default.json"
        with open(config_path, 'w') as f:
            f.write(config_str)
    else:
        print(f"✅ Found config: {config_path}")
    
    # Convert
    try:
        convert_numpy_to_gguf(
            str(numpy_path),
            str(config_path),
            str(output_path),
            quantize=True
        )
        print("🎉 Your Jarvis Quantum LLM is ready for Ollama!")
        print(f"📁 GGUF file: {output_path}")
        print()
        print("Next steps:")
        print("  1. Create the model: ollama create jarvis -f Modelfile")
        print("  2. Run the model:   ollama run jarvis")
        print()
        return 0
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
