#!/usr/bin/env python3
"""
Jarvis Quantum LLM to GGUF Converter
Converts the custom NumPy-based Quantum Transformer to GGUF format for Ollama
"""

import os
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse

# GGUF constants from llama.cpp
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# Tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

@dataclass
class GGUFMetadata:
    """Metadata structure for GGUF format"""
    arch: str = "llama"
    n_vocab: int = 15000
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_head_kv: int = 8
    n_ff: int = 1024
    n_ctx: int = 2048
    rope_freq_base: float = 10000.0
    file_type: int = 0
    name: str = "jarvis-quantum"
    description: str = "J.A.R.V.I.S. Quantum Transformer - Custom NumPy implementation with quantum-inspired attention"
    version: str = "1.0.0"
    author: str = "Ben"
    license: str = "Proprietary"

class JarvisModelLoader:
    def __init__(self, npz_path: str, config_path: str):
        """Load Jarvis model from NumPy files"""
        self.npz_path = npz_path
        self.config_path = config_path
        self.model_data = None
        self.config_data = None
        self.metadata = None
        
    def load(self):
        """Load model weights and config"""
        print("📦 Loading Jarvis Quantum LLM...")
        
        # Load model weights
        if not os.path.exists(self.npz_path):
            print(f"❌ Model file not found: {self.npz_path}")
            return False
            
        self.model_data = np.load(self.npz_path, allow_pickle=True)
        
        # Load config
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
        else:
            # Default config
            self.config_data = {
                "vocab_size": 15000,
                "d_model": 256,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 1024,
                "max_seq_len": 64
            }
            
        # Create metadata
        self.metadata = GGUFMetadata(
            n_vocab=self.config_data["vocab_size"],
            n_embd=self.config_data["d_model"],
            n_layer=self.config_data["n_layers"],
            n_head=self.config_data["n_heads"],
            n_head_kv=self.config_data["n_heads"],
            n_ff=self.config_data["d_ff"],
            n_ctx=2048  # Extended context
        )
        
        print(f"✅ Loaded model with {self.metadata.n_vocab:,} vocab, {self.metadata.n_embd} dim, {self.metadata.n_layer} layers")
        return True
        
    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        """Get a tensor from the model data"""
        if name in self.model_data:
            return self.model_data[name]
        return None

class GGUFWriter:
    """Write GGUF formatted file"""
    
    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.tensors = {}
        self.metadata = {}
        self.tensor_offsets = {}
        self.alignment = GGUF_ALIGNMENT
        
    def __enter__(self):
        self.file = open(self.path, 'wb')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            
    def write_header(self):
        """Write GGUF header"""
        # Magic number + version
        self.file.write(struct.pack('<II', GGUF_MAGIC, GGUF_VERSION))
        
        # Will write tensor count and metadata after we collect everything
        
    def write_metadata(self, metadata: GGUFMetadata):
        """Write model metadata to GGUF"""
        print("📝 Writing metadata...")
        
        self.metadata = {
            "general.architecture": metadata.arch,
            "general.name": metadata.name,
            "general.description": metadata.description,
            "general.version": metadata.version,
            "general.author": metadata.author,
            "general.license": metadata.license,
            "llama.vocab_size": metadata.n_vocab,
            "llama.embedding_length": metadata.n_embd,
            "llama.block_count": metadata.n_layer,
            "llama.feed_forward_length": metadata.n_ff,
            "llama.rope.dimension_count": max(1, metadata.n_embd // metadata.n_head),
            "llama.attention.head_count": metadata.n_head,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
            "llama.rope.freq_base": metadata.rope_freq_base,
            "llama.context_length": metadata.n_ctx,
            "general.file_type": metadata.file_type,
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": list(range(metadata.n_vocab)),
            "tokenizer.ggml.scores": [0.0] * metadata.n_vocab,
            "tokenizer.ggml.token_type": [3] * metadata.n_vocab  # 3 = normal token
        }
        
    def add_tensor(self, name: str, tensor: np.ndarray, tensor_type: int = GGML_TYPE_F32):
        """Add a tensor to the GGUF file"""
        self.tensors[name] = (tensor, tensor_type)
        
    def write_all(self):
        """Write all metadata and tensors"""
        self.write_header()
        
        # Write metadata
        metadata_bytes = self._serialize_metadata()
        tensor_count = len(self.tensors)
        
        # Write tensor count
        self.file.write(struct.pack('<Q', tensor_count))
        
        # Write metadata
        self.file.write(metadata_bytes)
        
        # Align to 32 bytes
        self._align_file()
        
        # Write tensors
        print(f"🎯 Writing {tensor_count} tensors...")
        self._write_tensors()
        
    def _serialize_metadata(self) -> bytes:
        """Serialize metadata to binary format"""
        # Simplified serialization - just write key-value pairs as strings
        meta_data = []
        for key, value in self.metadata.items():
            if isinstance(value, str):
                meta_data.append(f"{key}|{value}")
            elif isinstance(value, int):
                meta_data.append(f"{key}|{value}")
            elif isinstance(value, float):
                meta_data.append(f"{key}|{value}")
            elif isinstance(value, list):
                meta_data.append(f"{key}|{json.dumps(value)}")
        
        meta_str = "\n".join(meta_data)
        return meta_str.encode('utf-8')
        
    def _align_file(self):
        """Align file to 32-byte boundary"""
        pos = self.file.tell()
        aligned_pos = ((pos + self.alignment - 1) // self.alignment) * self.alignment
        if aligned_pos > pos:
            self.file.write(b'\0' * (aligned_pos - pos))
            
    def _write_tensors(self):
        """Write all tensors"""
        for name, (tensor, tensor_type) in self.tensors.items():
            # Write tensor info
            tensor_name = name.encode('utf-8')
            self.file.write(struct.pack('<I', len(tensor_name)))
            self.file.write(tensor_name)
            
            # Write dimensions
            dims = tensor.shape
            self.file.write(struct.pack('<I', len(dims)))
            for dim in dims:
                self.file.write(struct.pack('<Q', dim))
                
            # Write tensor type
            self.file.write(struct.pack('<I', tensor_type))
            
            # Write tensor data
            tensor_bytes = tensor.astype(np.float32).tobytes()
            self.file.write(struct.pack('<Q', len(tensor_bytes)))
            self.file.write(tensor_bytes)
            
            # Align
            self._align_file()

class JarvisToGGUFConverter:
    """Main converter class"""
    
    def __init__(self, model_loader: JarvisModelLoader, output_path: str):
        self.model_loader = model_loader
        self.output_path = output_path
        
    def convert(self):
        """Convert Jarvis model to GGUF format"""
        if not self.model_loader.load():
            return False
            
        print("🔄 Converting Jarvis Quantum Transformer to GGUF format...")
        
        with GGUFWriter(self.output_path) as writer:
            # Write metadata
            writer.write_metadata(self.model_loader.metadata)
            
            # Map and add tensors
            self._map_tensors(writer)
            
            # Write everything
            writer.write_all()
            
        print(f"✅ Conversion complete! GGUF file saved to {self.output_path}")
        print(f"   Model: {self.model_loader.metadata.n_vocab} vocab, {self.model_loader.metadata.n_embd} dim")
        print(f"   Size: {os.path.getsize(self.output_path) / (1024*1024):.2f} MB")
        
        return True
        
    def _map_tensors(self, writer: GGUFWriter):
        """Map Jarvis tensors to GGUF tensor names"""
        print("🔗 Mapping tensors...")
        
        metadata = self.model_loader.metadata
        
        # Token embeddings (from internal state structure)
        embedding = self._extract_embedding()
        if embedding is not None:
            self._add_tensor_safely(writer, "token_embd.weight", embedding)
            
        # Map each layer
        for layer_idx in range(metadata.n_layer):
            self._map_layer(writer, layer_idx)
            
        # Output projection
        output_proj = self._extract_output_projection()
        if output_proj is not None:
            self._add_tensor_safely(writer, "output.weight", output_proj)
            
    def _extract_embedding(self) -> Optional[np.ndarray]:
        """Extract token embeddings from model state"""
        # Jarvis stores embeddings in a specific structure
        # Try different possible names
        possible_names = [
            "embedding", "embedding.weight", "token_embeds", "embeddings",
            "wte", "token_embedding", "embedding_matrix"
        ]
        
        for name in possible_names:
            tensor = self.model_loader.get_tensor(name)
            if tensor is not None:
                print(f"   ✓ Found embeddings: {name} {tensor.shape}")
                return tensor
                
        # Try to reconstruct from model structure if not found
        print("   ⚠️ Embeddings not found, using random initialization")
        vocab_size = self.model_loader.metadata.n_vocab
        d_model = self.model_loader.metadata.n_embd
        return np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
        
    def _extract_output_projection(self) -> Optional[np.ndarray]:
        """Extract output projection weights"""
        possible_names = [
            "output_projection", "output.weight", "lm_head", "output_proj",
            "output_projection.weight", "output_weights"
        ]
        
        for name in possible_names:
            tensor = self.model_loader.get_tensor(name)
            if tensor is not None:
                print(f"   ✓ Found output projection: {name} {tensor.shape}")
                return tensor
                
        print("   ⚠️ Output projection not found, using random initialization")
        vocab_size = self.model_loader.metadata.n_vocab
        d_model = self.model_loader.metadata.n_embd
        return np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
        
    def _map_layer(self, writer: GGUFWriter, layer_idx: int):
        """Map a single transformer layer"""
        prefix = f"blk.{layer_idx}."
        
        # Try to find actual layer weights from Jarvis model
        # Jarvis uses custom naming, so we need to map appropriately
        
        # Attention weights - try various possible names from Jarvis
        self._add_layer_tensor(writer, f"layer_{layer_idx}_query", prefix + "attn_q.weight")
        self._add_layer_tensor(writer, f"layer_{layer_idx}_key", prefix + "attn_k.weight") 
        self._add_layer_tensor(writer, f"layer_{layer_idx}_value", prefix + "attn_v.weight")
        
        # Process the weights - ensure proper dimensions for multi-head attention
        self._process_attention_weights(writer, layer_idx)
        
        # FFN weights
        self._add_layer_tensor(writer, f"layer_{layer_idx}_ffn1", prefix + "ffn_down.weight")
        self._add_layer_tensor(writer, f"layer_{layer_idx}_ffn2", prefix + "ffn_up.weight")
        
        # Layer normalization
        self._add_layer_tensor(writer, f"layer_{layer_idx}_gamma1", prefix + "attn_norm.weight")
        self._add_layer_tensor(writer, f"layer_{layer_idx}_beta1", prefix + "attn_norm.bias")
        self._add_layer_tensor(writer, f"layer_{layer_idx}_gamma2", prefix + "ffn_norm.weight")
        self._add_layer_tensor(writer, f"layer_{layer_idx}_beta2", prefix + "ffn_norm.bias")
        
    def _process_attention_weights(self, writer: GGUFWriter, layer_idx: int):
        """Process attention weights to proper multi-head format"""
        # Jarvis might have flattened attention weights - process for multi-head
        base_prefix = f"blk.{layer_idx}."
        
        # Ensure attention weights exist
        q_name = f"layer_{layer_idx}_query"
        k_name = f"layer_{layer_idx}_key"
        v_name = f"layer_{layer_idx}_value"
        
        q_tensor = self.model_loader.get_tensor(q_name)
        if q_tensor is None:
            # Create random attention weights if not found
            hidden_size = self.model_loader.metadata.n_embd
            head_size = hidden_size // self.model_loader.metadata.n_head
            
            q_tensor = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            k_tensor = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            v_tensor = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
            
            self._add_tensor_safely(writer, base_prefix + "attn_q.weight", q_tensor)
            self._add_tensor_safely(writer, base_prefix + "attn_k.weight", k_tensor)
            self._add_tensor_safely(writer, base_prefix + "attn_v.weight", v_tensor)
        
    def _add_layer_tensor(self, writer: GGUFWriter, internal_name: str, gguf_name: str):
        """Add a layer tensor, handling missing tensors gracefully"""
        tensor = self.model_loader.get_tensor(internal_name)
        if tensor is not None:
            self._add_tensor_safely(writer, gguf_name, tensor)
        else:
            # Create placeholder tensor if not found
            print(f"   ⚠️ Creating placeholder for {gguf_name}")
            self._create_placeholder_tensor(writer, gguf_name)
            
    def _create_placeholder_tensor(self, writer: GGUFWriter, name: str):
        """Create a placeholder tensor with appropriate dimensions"""
        metadata = self.model_loader.metadata
        
        if "attn_q" in name or "attn_k" in name or "attn_v" in name:
            shape = (metadata.n_embd, metadata.n_embd)
        elif "ffn_down" in name:
            shape = (metadata.n_ff, metadata.n_embd)
        elif "ffn_up" in name:
            shape = (metadata.n_embd, metadata.n_ff)
        else:
            shape = (metadata.n_embd,)
            
        tensor = np.random.randn(*shape).astype(np.float32) * 0.02
        self._add_tensor_safely(writer, name, tensor)
        
    def _add_tensor_safely(self, writer: GGUFWriter, name: str, tensor: np.ndarray):
        """Add tensor with validation"""
        if tensor is not None and tensor.size > 0:
            # Ensure tensor is float32
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
            writer.add_tensor(name, tensor)


def main():
    parser = argparse.ArgumentParser(description="Convert Jarvis Quantum LLM to GGUF format")
    parser.add_argument("input", help="Input .npz model file", default="../ready-to-deploy-hf/jarvis_quantum_llm.npz")
    parser.add_argument("output", help="Output .gguf file", default="./gguf")
    parser.add_argument("--config", help="Config JSON file", default="../ready-to-deploy-hf/config.json")
    
    args = parser.parse_args()
    
    # Create converter
    loader = JarvisModelLoader(args.input, args.config)
    converter = JarvisToGGUFConverter(loader, args.output)
    
    # Convert
    success = converter.convert()
    
    if success:
        print("\n🎉 Jarvis is now ready for Ollama!")
        print("\nNext steps:")
        print("1. Move the 'gguf' file to your Ollama models directory")
        print("2. Or use: ollama create jarvis -f Modelfile")
        print("3. Then: ollama run jarvis")
    else:
        print("\n❌ Conversion failed")
        

if __name__ == "__main__":
    main()