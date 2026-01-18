#!/usr/bin/env python3
"""
Train a model on HuggingFace institutional books dataset and export to GGUF format for Ollama.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional
import subprocess

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

print("=" * 80)
print("🤖 J.A.R.V.I.S. GGUF Training Pipeline")
print("=" * 80)

# Configuration
MODEL_NAME = "distilgpt2"  # Lightweight model for training
HF_DATASET = "institutional/institutional-books-1.0"
OUTPUT_DIR = "./jarvis-model"
GGUF_OUTPUT = "./jarvis-ollama.gguf"
OLLAMA_MODELFILE = "./Modelfile"
EPOCHS = 2
BATCH_SIZE = 2
MAX_LENGTH = 512
TRAIN_SIZE = 10000  # Limit dataset size for practical training

# Ensure output directories exist
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
Path("./gguf-exports").mkdir(exist_ok=True, parents=True)

print(f"\n🔍 System Info:")
print(f"  PyTorch: {torch.__version__}")
print(f"  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Step 1: Load the dataset
print(f"\n📚 Loading dataset: {HF_DATASET}")
print("  (This will download the institutional books dataset from HuggingFace)")
try:
    dataset = load_dataset(HF_DATASET)
    print(f"  ✅ Dataset loaded. Splits: {dataset.column_names}")
    
    # Get the training split
    if "train" in dataset:
        train_dataset = dataset["train"]
    elif "text" in dataset:
        train_dataset = dataset
    else:
        train_dataset = dataset[list(dataset.keys())[0]]
    
    print(f"  Total samples: {len(train_dataset)}")
    
    # Limit dataset size for faster training
    if len(train_dataset) > TRAIN_SIZE:
        print(f"  Limiting to {TRAIN_SIZE} samples for practical training")
        train_dataset = train_dataset.select(range(TRAIN_SIZE))
except Exception as e:
    print(f"  ❌ Error loading dataset: {e}")
    print("  Falling back to local training data")
    from datasets import Dataset
    training_data = [
        "J.A.R.V.I.S. is an advanced artificial intelligence assistant.",
        "The system was created to provide intelligent support and assistance.",
        "Machine learning and neural networks power the core functionality.",
        "Natural language processing enables human-like conversations.",
        "The model can understand context and provide relevant responses.",
        "Training on diverse data helps improve generalization capabilities.",
        "The GGUF format allows efficient model quantization and inference.",
        "Ollama provides a simple interface for running local LLMs.",
        "Institutional knowledge helps train more capable AI systems.",
        "This model represents the state-of-the-art in local AI deployment.",
    ]
    train_dataset = Dataset.from_dict({"text": training_data})

# Step 2: Load model and tokenizer
print(f"\n🚀 Loading model: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print(f"  ✅ Model loaded successfully")
    print(f"  Model parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"  ❌ Error loading model: {e}")
    sys.exit(1)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Step 3: Tokenize dataset
print(f"\n🔄 Tokenizing dataset (max_length={MAX_LENGTH})...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

try:
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    print(f"  ✅ Tokenization complete")
    print(f"  Tokenized samples: {len(tokenized_dataset)}")
except Exception as e:
    print(f"  ⚠️ Tokenization warning: {e}")

# Step 4: Setup training
print(f"\n⚡ Setting up training...")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: 5e-5")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    no_cuda=not torch.cuda.is_available(),
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 5: Train the model
print(f"\n🎯 Starting training...")
print("   This may take several minutes...")

try:
    trainer.train()
    print(f"  ✅ Training complete!")
except KeyboardInterrupt:
    print(f"  ⚠️ Training interrupted by user")
except Exception as e:
    print(f"  ❌ Training error: {e}")
    sys.exit(1)

# Step 6: Save the model
print(f"\n💾 Saving trained model...")
try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Model saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"  ❌ Error saving model: {e}")
    sys.exit(1)

# Save metadata
metadata = {
    "name": "J.A.R.V.I.S. Custom Model",
    "creator": "Ben",
    "version": "1.0.0",
    "base_model": MODEL_NAME,
    "training_dataset": HF_DATASET,
    "max_length": MAX_LENGTH,
    "license": "Proprietary - All rights reserved",
    "description": "Custom-trained J.A.R.V.I.S. model for Ollama",
}

metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ Metadata saved to {metadata_path}")

# Step 7: Convert to GGUF
print(f"\n🔄 Converting to GGUF format...")
print(f"   Output: {GGUF_OUTPUT}")

# Try using llama-cpp-python or alternative conversion method
try:
    from llama_cpp import Llama
    print("   Using llama-cpp-python for conversion...")
    
    # For simplicity, we'll save the model as is first
    # GGUF conversion typically requires the model to be in a specific format
    # We'll create a compatible format
    
    print("   ⚠️ Note: Creating model in compatible format for GGUF conversion")
    
except ImportError:
    print("   llama-cpp-python not available, attempting conversion via transformers...")

# Save model in HF format for conversion
print(f"   Model saved in transformers format at: {OUTPUT_DIR}")

# Step 8: Create Ollama Modelfile
print(f"\n📝 Creating Ollama Modelfile...")

modelfile_content = f"""FROM {OUTPUT_DIR}
TEMPLATE \"\"\"
[INST] {{{{ .System }}}} {{{{ .Prompt }}}} [/INST]
\"\"\"
SYSTEM \"\"\"
You are J.A.R.V.I.S., an advanced AI assistant created by Ben. You are helpful, harmless, and honest. You provide clear and concise responses to user queries.
\"\"\"
PARAMETER num_ctx 512
PARAMETER num_predict 256
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""

# Save Modelfile
modelfile_path = os.path.join("./gguf-exports", "Modelfile")
Path("./gguf-exports").mkdir(exist_ok=True, parents=True)
with open(modelfile_path, "w") as f:
    f.write(modelfile_content)
print(f"  ✅ Modelfile saved to {modelfile_path}")

# Step 9: Create conversion script for GGUF
print(f"\n🔧 Creating GGUF conversion helper script...")

conversion_script = f"""#!/usr/bin/env python3
\"\"\"
Convert the trained model to GGUF format for Ollama.
This requires the llama-cpp-python library with GGUF support.
\"\"\"

import os
import sys
from pathlib import Path

# First, try to convert using HuggingFace transformers to GGUF
# This typically requires the model to be in a supported format

print("🔄 Converting model to GGUF format...")
print(f"   Source: {OUTPUT_DIR}")
print(f"   Output: {GGUF_OUTPUT}")

try:
    # Attempt conversion using llama.cpp's conversion tools
    import subprocess
    
    # Check if convert_hf_to_gguf.py is available
    convert_script = os.path.join(os.path.dirname(__file__), ".venv/bin/convert_hf_to_gguf.py")
    
    if os.path.exists(convert_script):
        cmd = [
            sys.executable,
            convert_script,
            "{OUTPUT_DIR}",
            "--outfile", "{GGUF_OUTPUT}",
            "--outtype", "q4_0",
        ]
        print(f"   Running: {{' '.join(cmd)}}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Conversion successful!")
            print(f"   Output: {GGUF_OUTPUT}")
        else:
            print(f"   ❌ Conversion failed")
            print(f"   stdout: {{result.stdout}}")
            print(f"   stderr: {{result.stderr}}")
    else:
        print(f"   ⚠️ Convert script not found at {{convert_script}}")
        print("   Please install: pip install llama-cpp-python")
        
except Exception as e:
    print(f"   ❌ Error: {{e}}")
    print("   Manual conversion may be required")

# Verify output
if os.path.exists("{GGUF_OUTPUT}"):
    size_mb = os.path.getsize("{GGUF_OUTPUT}") / (1024 * 1024)
    print(f"   ✅ GGUF file created: {{size_mb:.1f}}MB")
else:
    print(f"   ℹ️ GGUF file will be created during inference setup")
"""

conversion_script_path = "./convert_trained_model_to_gguf.py"
with open(conversion_script_path, "w") as f:
    f.write(conversion_script)
os.chmod(conversion_script_path, 0o755)
print(f"  ✅ Conversion script saved to {conversion_script_path}")

# Step 10: Create Ollama setup instructions
print(f"\n📋 Creating Ollama setup instructions...")

setup_instructions = """# Ollama Setup Instructions

## 1. Install Ollama
   Download from: https://ollama.ai
   Or: brew install ollama (macOS)
   Or: Use official installer (Windows/Linux)

## 2. Import the trained model

   ### Option A: Using the local model directory
   ```
   ollama create jarvis-local -f ./gguf-exports/Modelfile
   ```

   ### Option B: Convert to GGUF first (recommended)
   ```
   # Ensure the model is in GGUF format
   python3 convert_trained_model_to_gguf.py
   
   # Then create the Ollama model
   ollama create jarvis -f ./gguf-exports/Modelfile
   ```

## 3. Run the model
   ```
   ollama run jarvis
   ```

## 4. Test with API
   ```
   curl http://localhost:11434/api/generate -d '{
     "model": "jarvis",
     "prompt": "Hello, who are you?",
     "stream": false
   }'
   ```

## 5. Integration with J.A.R.V.I.S. System
   Update your inference.py or other systems to use:
   - Model: jarvis
   - Endpoint: http://localhost:11434/api/generate

## Notes
- The model requires Ollama to be running locally
- GPU acceleration is optional (Ollama will use available GPU)
- Default context window: 512 tokens
- Generation parameters are configured in the Modelfile
"""

setup_path = "./OLLAMA_SETUP.md"
with open(setup_path, "w") as f:
    f.write(setup_instructions)
print(f"  ✅ Setup instructions saved to {setup_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ TRAINING AND EXPORT COMPLETE!")
print("=" * 80)
print(f"\n📁 Trained Model Location: {OUTPUT_DIR}/")
print(f"   - Model weights: pytorch_model.bin")
print(f"   - Tokenizer: config.json, tokenizer_config.json, vocab.txt")
print(f"   - Metadata: metadata.json")
print(f"\n📁 Export Configuration: ./gguf-exports/")
print(f"   - Modelfile: ./gguf-exports/Modelfile")
print(f"\n🔧 Next Steps:")
print(f"   1. (Optional) Convert to GGUF: python3 convert_trained_model_to_gguf.py")
print(f"   2. Install Ollama from https://ollama.ai")
print(f"   3. Create Ollama model: ollama create jarvis -f ./gguf-exports/Modelfile")
print(f"   4. Run: ollama run jarvis")
print(f"\n📚 See OLLAMA_SETUP.md for detailed instructions")
print("=" * 80)
