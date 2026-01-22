#!/usr/bin/env python3
"""
Jarvis Quantum LLM - Advanced Quantization Script
Try different quantization methods for size/speed/quality tradeoffs
"""

import argparse
import struct
import json
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np


GGUF_MAGIC = 0x47475546
GGUF_VERSION = 3


class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9


def quantize_q4_0(weights: np.ndarray) -> bytes:
    """4-bit quantization - smallest size, fastest, lower quality"""
    weights_flat = weights.flatten().astype(np.float32)
    block_size = 32
    n_blocks = (len(weights_flat) + block_size - 1) // block_size
    
    quantized = bytearray()
    
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(weights_flat))
        block = weights_flat[start:end]
        
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)), mode='constant')
        
        max_val = np.max(np.abs(block))
        scale = max_val / 7.0 if max_val > 0 else 1.0
        
        quantized_block = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
        
        # Pack two 4-bit values per byte
        packed = bytearray()
        for j in range(0, len(quantized_block), 2):
            val1 = quantized_block[j] & 0x0F
            val2 = quantized_block[j+1] & 0x0F if j+1 < len(quantized_block) else 0
            packed.append((val2 << 4) | val1)
        
        quantized.extend(struct.pack('e', scale))
        quantized.extend(packed)
    
    return bytes(quantized)


def quantize_q8_0(weights: np.ndarray) -> bytes:
    """8-bit quantization - good balance"""
    weights_flat = weights.flatten().astype(np.float32)
    block_size = 32
    n_blocks = (len(weights_flat) + block_size - 1) // block_size
    
    quantized = bytearray()
    
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(weights_flat))
        block = weights_flat[start:end]
        
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)), mode='constant')
        
        max_val = np.max(np.abs(block))
        scale = max_val / 127.0 if max_val > 0 else 1.0
        
        quantized_block = np.round(block / scale).astype(np.int8)
        
        quantized.extend(struct.pack('e', scale))
        quantized.extend(quantized_block.tobytes())
    
    return bytes(quantized)


def quantize_f16(weights: np.ndarray) -> bytes:
    """16-bit float - smaller than F32, good quality"""
    return weights.astype(np.float16).tobytes()


def quantize_f32(weights: np.ndarray) -> bytes:
    """32-bit float - full precision, largest size"""
    return weights.astype(np.float32).tobytes()


def write_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('Q', len(encoded)))
    f.write(encoded)


def write_tensor(f, name: str, tensor: np.ndarray, quant_type: str):
    """Write tensor with specified quantization"""
    write_string(f, name)
    
    n_dims = len(tensor.shape)
    f.write(struct.pack('I', n_dims))
    
    for dim in reversed(tensor.shape):
        f.write(struct.pack('Q', dim))
    
    # Choose quantization
    if quant_type == "q4_0" and tensor.size > 1000:
        f.write(struct.pack('I', GGMLType.Q4_0))
        data = quantize_q4_0(tensor)
    elif quant_type == "q8_0" and tensor.size > 1000:
        f.write(struct.pack('I', GGMLType.Q8_0))
        data = quantize_q8_0(tensor)
    elif quant_type == "f16":
        f.write(struct.pack('I', GGMLType.F16))
        data = quantize_f16(tensor)
    else:  # f32
        f.write(struct.pack('I', GGMLType.F32))
        data = quantize_f32(tensor)
    
    f.write(data)
    size_kb = len(data) / 1024
    print(f"  ✓ {name}: {tensor.shape} ({size_kb:.1f} KB)")


def convert_with_quantization(
    numpy_path: str,
    config_path: str,
    output_path: str,
    quant_type: str
):
    """Convert with specified quantization"""
    print("\n🔢 JARVIS QUANTUM LLM - Quantization Tool")
    print("=" * 60)
    print(f"📁 Input:  {numpy_path}")
    print(f"💾 Output: {output_path}")
    print(f"🔧 Quantization: {quant_type.upper()}")
    print()
    
    # Load data
    data = np.load(numpy_path)
    with open(config_path) as f:
        config = json.load(f)
    
    total_params = sum(data[k].size for k in data.keys())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Write GGUF
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', GGUF_MAGIC))
        f.write(struct.pack('I', GGUF_VERSION))
        
        metadata = {
            "general.architecture": "jarvis-quantum",
            "general.name": "Jarvis Quantum LLM",
            "general.quantization": quant_type.upper(),
            "jarvis.vocab_size": str(config["vocab_size"]),
            "jarvis.d_model": str(config["d_model"]),
            "jarvis.n_layers": str(config["n_layers"]),
            "jarvis.n_heads": str(config["n_heads"]),
        }
        
        f.write(struct.pack('Q', len(metadata)))
        for key, value in metadata.items():
            write_string(f, key)
            f.write(struct.pack('I', 4))  # string type
            write_string(f, value)
        
        # Write tensors
        n_tensors = len([k for k in data.keys() if k not in ['vocab_size', 'd_model', 'n_layers', 'n_heads', 'd_ff']])
        f.write(struct.pack('Q', n_tensors))
        
        print(f"\n✍️  Writing {n_tensors} tensors...")
        for key in sorted(data.keys()):
            if key not in ['vocab_size', 'd_model', 'n_layers', 'n_heads', 'd_ff']:
                write_tensor(f, key, data[key], quant_type)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Quantization complete!")
    print(f"📊 Output size: {size_mb:.2f} MB")
    
    # Show comparison
    if quant_type == "q4_0":
        print(f"💡 ~75% smaller than F32, fastest inference")
    elif quant_type == "q8_0":
        print(f"💡 ~50% smaller than F32, good balance")
    elif quant_type == "f16":
        print(f"💡 ~50% smaller than F32, high quality")
    else:
        print(f"💡 Full precision, largest size")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Jarvis Quantum LLM for Ollama"
    )
    parser.add_argument(
        "--quant",
        choices=["q4_0", "q8_0", "f16", "f32"],
        default="q8_0",
        help="Quantization type (default: q8_0)"
    )
    parser.add_argument(
        "--input",
        help="Input NumPy file path"
    )
    parser.add_argument(
        "--output",
        help="Output GGUF file path"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    numpy_path = args.input or str(project_root / "ready-to-deploy-hf" / "jarvis_quantum_llm.npz")
    config_path = str(project_root / "ready-to-deploy-hf" / "config.json")
    output_path = args.output or str(script_dir / f"jarvis-quantum-{args.quant}.gguf")
    
    if not Path(numpy_path).exists():
        print(f"❌ Error: {numpy_path} not found!")
        return 1
    
    try:
        convert_with_quantization(numpy_path, config_path, output_path, args.quant)
        
        print("Next steps:")
        print(f"  1. Update Modelfile to use: FROM ./{Path(output_path).name}")
        print(f"  2. Create model: ollama create jarvis -f Modelfile")
        print(f"  3. Run: ollama run jarvis")
        print()
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
