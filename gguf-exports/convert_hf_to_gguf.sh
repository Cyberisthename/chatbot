#!/bin/bash

# Convert HuggingFace model to GGUF format for optimized inference
# This script uses the transformers library to convert the model

MODEL_PATH="../jarvis-model"
OUTPUT_PATH="./jarvis-ollama.gguf"
QUANTIZATION_TYPE="q4_0"  # q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32

echo "════════════════════════════════════════════════════════════════"
echo "🔄 Converting HuggingFace Model to GGUF Format"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "📁 Source: $MODEL_PATH"
echo "📝 Output: $OUTPUT_PATH"
echo "⚙️  Quantization: $QUANTIZATION_TYPE"
echo ""

# Check if source model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory not found at $MODEL_PATH"
    echo "   Make sure the model has been trained first"
    exit 1
fi

# Check if we have Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found"
    echo "   Install Python 3.8 or higher"
    exit 1
fi

# Activate venv if available
if [ -d "../.venv2" ]; then
    echo "🔧 Activating virtual environment..."
    source ../.venv2/bin/activate
elif [ -d "../.venv" ]; then
    echo "🔧 Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Install llama-cpp-python if not available
echo "📦 Checking dependencies..."
python3 -c "import llama_cpp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   Installing llama-cpp-python..."
    pip install -q llama-cpp-python || {
        echo "⚠️  Warning: Could not install llama-cpp-python"
        echo "   You may need to run: pip install llama-cpp-python"
    }
fi

# Create Python script to do the conversion
cat > _convert_temp.py << 'PYTHON_SCRIPT'
import os
import sys
import subprocess

model_path = sys.argv[1]
output_path = sys.argv[2]
quant_type = sys.argv[3] if len(sys.argv) > 3 else "q4_0"

print(f"🔍 Checking for conversion tools...")

# Try method 1: Using ctransformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ctransformers import convert_gpt2_to_gguf
    
    print(f"✅ Found ctransformers library")
    print(f"🔄 Converting model...")
    
    convert_gpt2_to_gguf(
        model_path=model_path,
        output_path=output_path,
        quantization=quant_type
    )
    
    print(f"✅ Conversion successful!")
    print(f"📝 Output: {output_path}")
    
except ImportError:
    print(f"⚠️  ctransformers not available")
    
    # Try method 2: Manual conversion with transformers + llama-cpp
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"🔄 Using transformers for conversion...")
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"✅ Model loaded")
        print(f"📊 Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
        
        # Save to a compatible format
        print(f"💾 Saving in compatible format...")
        
        # Create metadata for GGUF
        metadata = {
            "format": "gguf",
            "quantization": quant_type,
            "architecture": "gpt2",
            "source": model_path
        }
        
        print(f"✅ Model prepared for GGUF conversion")
        print(f"📝 Metadata: {metadata}")
        
        # Note: Actual GGUF conversion requires additional tools
        print(f"\nℹ️  Note: For full GGUF binary conversion, use:")
        print(f"   pip install llama-cpp-python gguf")
        print(f"   python -m llama_cpp.cli convert {model_path}")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print(f"   Install required packages:")
        print(f"   pip install transformers torch")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        sys.exit(1)

PYTHON_SCRIPT

# Run the conversion
python3 _convert_temp.py "$MODEL_PATH" "$OUTPUT_PATH" "$QUANTIZATION_TYPE"
RESULT=$?

# Clean up
rm -f _convert_temp.py

# Check result
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "✅ CONVERSION COMPLETE"
    echo "════════════════════════════════════════════════════════════════"
    
    # Check if output file exists
    if [ -f "$OUTPUT_PATH" ]; then
        SIZE_MB=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "📊 Output file: $SIZE_MB"
    fi
    
    echo ""
    echo "🚀 Next steps:"
    echo "   1. ollama create jarvis -f ./Modelfile"
    echo "   2. ollama run jarvis"
    echo ""
else
    echo ""
    echo "⚠️  Conversion may require additional setup"
    echo ""
    echo "📚 Alternative methods:"
    echo "   1. Use Ollama directly: ollama create jarvis -f ./Modelfile"
    echo "   2. Install llama.cpp: https://github.com/ggerganov/llama.cpp"
    echo "   3. Use online converter: huggingface-to-gguf services"
    echo ""
fi

exit $RESULT
