#!/usr/bin/env python3
"""
GGUF File Validator for Jarvis
Verifies the integrity and structure of the exported GGUF file
"""

import json
import struct
import sys
from pathlib import Path


def validate_gguf(file_path: str):
    """Validate GGUF file structure"""
    print(f"🔍 Validating GGUF file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            # Check magic number
            magic = f.read(4)
            if magic != b'GGUF':
                print(f"❌ Invalid magic number: {magic}")
                return False
            print("✅ Magic number valid (GGUF)")
            
            # Read version
            version_data = f.read(4)
            version = struct.unpack('<I', version_data)[0]
            print(f"✅ Version: {version}")
            
            # Read tensor count
            tensor_count_data = f.read(8)
            tensor_count = struct.unpack('<Q', tensor_count_data)[0]
            print(f"✅ Tensor count: {tensor_count}")
            
            # Validate each tensor
            for i in range(tensor_count):
                print(f"\n📦 Tensor {i + 1}:")
                
                # Read tensor name
                name_len_data = f.read(4)
                if not name_len_data:
                    print("❌ Unexpected end of file reading name length")
                    return False
                name_len = struct.unpack('<I', name_len_data)[0]
                
                name_data = f.read(name_len)
                name = name_data.decode('utf-8')
                print(f"  Name: {name}")
                
                # Read dimensions
                dims_count_data = f.read(4)
                dims_count = struct.unpack('<I', dims_count_data)[0]
                print(f"  Dimensions: {dims_count}")
                
                total_size = 1
                for j in range(dims_count):
                    dim_data = f.read(8)
                    dim = struct.unpack('<Q', dim_data)[0]
                    print(f"    Dim {j + 1}: {dim}")
                    total_size *= dim
                
                # Read data type
                dtype_data = f.read(4)
                dtype = struct.unpack('<I', dtype_data)[0]
                print(f"  Data type: {dtype}")
                
                # Read data size
                data_size_data = f.read(8)
                data_size = struct.unpack('<Q', data_size_data)[0]
                print(f"  Data size: {data_size} bytes")
                
                # Skip actual data for now
                f.read(data_size)
                
                # Check alignment padding
                pos = f.tell()
                padding = 32 - (pos % 32)
                if padding < 32:
                    f.read(padding)
            
            # Check if file ends properly
            remaining = f.read()
            if remaining:
                print(f"⚠️  Extra data at end: {len(remaining)} bytes")
            else:
                print("✅ File ends cleanly")
        
        file_size = Path(file_path).stat().st_size
        print(f"\n📊 File size: {file_size / (1024 * 1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def extract_metadata(file_path: str):
    """Extract and display metadata from GGUF file"""
    print(f"\n📋 Extracting metadata from {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            # Skip header
            f.read(16)  # magic + version + tensor_count
            
            # Read tensors
            tensor_count_data = f.read(8)
            tensor_count = struct.unpack('<Q', tensor_count_data)[0]
            
            metadata = {}
            
            for i in range(tensor_count):
                # Read tensor name
                name_len_data = f.read(4)
                name_len = struct.unpack('<I', name_len_data)[0]
                name = f.read(name_len).decode('utf-8')
                
                # Skip to data
                dims_count = struct.unpack('<I', f.read(4))[0]
                for _ in range(dims_count):
                    f.read(8)
                
                f.read(4)  # dtype
                data_size = struct.unpack('<Q', f.read(8))[0]
                
                if name.endswith('.json'):
                    json_data = f.read(data_size)
                    try:
                        metadata = json.loads(json_data.decode('utf-8'))
                        print("✅ Metadata extracted successfully")
                    except:
                        print("⚠️ Could not parse metadata JSON")
                else:
                    f.read(data_size)
                
                # Skip padding
                pos = f.tell()
                padding = 32 - (pos % 32)
                if padding < 32:
                    f.read(padding)
            
            if metadata:
                print(f"\n🏷️  Model Information:")
                for key, value in metadata.items():
                    if key != 'training':  # Skip training details for brevity
                        print(f"  {key}: {value}")
                
                if 'training' in metadata:
                    print(f"\n🎓 Training Details:")
                    for key, value in metadata['training'].items():
                        print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting metadata: {e}")
        return False


def main():
    if len(sys.argv) > 1:
        gguf_file = sys.argv[1]
    else:
        gguf_file = "gguf"
    
    print("=" * 60)
    print("🔍 GGUF File Validator - Jarvis Quantum LLM")
    print("=" * 60)
    
    # Validate structure
    is_valid = validate_gguf(gguf_file)
    
    # Extract metadata
    if is_valid:
        extract_metadata(gguf_file)
    
    print("\n" + "=" * 60)
    if is_valid:
        print("✅ GGUF file validation PASSED")
        print("🎉 Jarvis is ready for Ollama!")
    else:
        print("❌ GGUF file validation FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()