#!/usr/bin/env python3
"""
Complete Training & GGUF Export Pipeline
=========================================
This script trains a model on quantum lab data and exports it to GGUF format.
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    missing = []
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import datasets
        print(f"  ✓ Datasets {datasets.__version__}")
    except ImportError:
        missing.append("datasets")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def train_model(
    data_path: str = "lab_training_data.jsonl",
    model_name: str = "distilgpt2",
    output_dir: str = "jarvis-lab-model",
    epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 512,
) -> bool:
    """Train a model on the quantum lab data."""
    
    print("\n" + "=" * 60)
    print("STEP 1: Training Model")
    print("=" * 60)
    
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from datasets import load_dataset
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Training data not found: {data_path}")
        print("Available data files:")
        for f in Path(".").glob("*.jsonl"):
            print(f"  - {f}")
        return False
    
    print(f"\n📂 Loading data from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    # Prepare text column
    def format_sample(example):
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        text = f"Instruction: {instruction}\n\nAnswer: {output}"
        return {"text": text}
    
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    
    print(f"\n🤖 Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    print("  ✓ Model loaded")
    
    print(f"\n🔄 Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    print("  ✓ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )
    
    print(f"\n🏋️  Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False
    
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "name": "Jarvis Lab Quantum Model",
        "creator": "Ben Lab",
        "version": "1.0.0",
        "base_model": model_name,
        "training_samples": len(dataset),
        "epochs": epochs,
        "description": "Fine-tuned on quantum phase experiments and lab data"
    }
    
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("  ✓ Model saved successfully")
    return True


def convert_to_gguf(model_dir: str = "jarvis-lab-model", output_path: Optional[str] = None) -> bool:
    """Convert the trained model to GGUF format using llama.cpp."""
    
    print("\n" + "=" * 60)
    print("STEP 2: Converting to GGUF Format")
    print("=" * 60)
    
    if not Path(model_dir).exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Clone llama.cpp if needed
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        print("\n📦 Cloning llama.cpp repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                check=True,
                capture_output=True,
            )
            print("  ✓ llama.cpp cloned")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone llama.cpp: {e}")
            return False
    
    # Install llama.cpp requirements
    requirements_file = llama_cpp_dir / "requirements.txt"
    if requirements_file.exists():
        print("\n📦 Installing llama.cpp requirements...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                check=True,
            )
            print("  ✓ Requirements installed")
        except subprocess.CalledProcessError:
            print("  ⚠️  Some requirements may not have installed")
    
    # Convert to GGUF
    if output_path is None:
        output_path = f"{model_dir}.gguf"
    
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try older naming convention
        convert_script = llama_cpp_dir / "convert.py"
    
    if not convert_script.exists():
        print(f"❌ Conversion script not found in llama.cpp")
        print("Available Python files in llama.cpp:")
        for f in llama_cpp_dir.glob("*.py"):
            print(f"  - {f.name}")
        return False
    
    print(f"\n🔄 Converting {model_dir} to GGUF...")
    print(f"  Output: {output_path}")
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(convert_script),
                model_dir,
                "--outfile", output_path,
                "--outtype", "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(f"  ✓ Conversion complete")
        
        # Show file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  📊 File size: {file_size:.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    
    return True


def quantize_gguf(input_path: str, output_path: Optional[str] = None, quantization: str = "Q4_0") -> bool:
    """Quantize the GGUF model for smaller size and faster inference."""
    
    print("\n" + "=" * 60)
    print("STEP 3: Quantizing GGUF (Optional)")
    print("=" * 60)
    
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    llama_cpp_dir = Path("llama.cpp")
    
    # Build quantize tool if needed
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_dir / "llama-quantize"
    
    if not quantize_bin.exists():
        print("  ⚠️  llama-quantize binary not found, skipping quantization")
        print("  To enable quantization, build llama.cpp:")
        print("    cd llama.cpp && make llama-quantize")
        return True  # Not a failure, just skip
    
    if output_path is None:
        base = Path(input_path).stem
        output_path = f"{base}-{quantization.lower()}.gguf"
    
    print(f"\n🗜️  Quantizing to {quantization}...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    try:
        subprocess.run(
            [str(quantize_bin), input_path, output_path, quantization],
            check=True,
        )
        
        # Compare sizes
        input_size = Path(input_path).stat().st_size / (1024 * 1024)
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        compression = (1 - output_size / input_size) * 100
        
        print(f"  ✓ Quantization complete")
        print(f"  📊 Original: {input_size:.1f} MB")
        print(f"  📊 Quantized: {output_size:.1f} MB ({compression:.1f}% reduction)")
        
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Quantization failed: {e}")
        return False
    
    return True


def create_modelfile(gguf_path: str, model_name: str = "jarvis-lab") -> None:
    """Create an Ollama Modelfile for easy deployment."""
    
    print("\n" + "=" * 60)
    print("STEP 4: Creating Ollama Modelfile")
    print("=" * 60)
    
    modelfile_content = f"""FROM ./{gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM \"\"\"
You are Jarvis Lab AI, an expert in quantum phases and experimental physics.
You understand quantum phase transitions, time-reversal instability (TRI),
symmetry-protected topological phases (SPT), and the Jarvis-5090X quantum simulator.
You provide clear, scientific explanations using Ben's lab terminology.
\"\"\"
"""
    
    modelfile_path = Path("Modelfile.jarvis-lab")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"  ✓ Modelfile created: {modelfile_path}")
    print()
    print("To create the Ollama model, run:")
    print(f"  ollama create {model_name} -f {modelfile_path}")
    print()
    print("Then chat with it:")
    print(f"  ollama run {model_name}")


def main():
    """Main training and export pipeline."""
    
    print()
    print("╔" + "=" * 58 + "╗")
    print("║  Jarvis Lab: Complete Training & GGUF Export Pipeline  ║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Configuration
    model_dir = "jarvis-lab-model"
    gguf_path = f"{model_dir}.gguf"
    
    # Step 1: Train model
    success = train_model(
        data_path="lab_training_data.jsonl",
        model_name="distilgpt2",
        output_dir=model_dir,
        epochs=3,
        batch_size=2,
    )
    
    if not success:
        print("\n❌ Training failed")
        sys.exit(1)
    
    # Step 2: Convert to GGUF
    success = convert_to_gguf(model_dir, gguf_path)
    
    if not success:
        print("\n❌ GGUF conversion failed")
        print("The trained model is still available in:", model_dir)
        sys.exit(1)
    
    # Step 3: Optional quantization
    quantize_gguf(gguf_path, f"{model_dir}-q4.gguf", "Q4_0")
    
    # Step 4: Create Modelfile
    create_modelfile(gguf_path)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE - Your Model is Ready!")
    print("=" * 60)
    print()
    print("📁 Files created:")
    print(f"  - {model_dir}/         (HuggingFace format)")
    print(f"  - {gguf_path}          (GGUF format)")
    if Path(f"{model_dir}-q4.gguf").exists():
        print(f"  - {model_dir}-q4.gguf  (Quantized GGUF)")
    print(f"  - Modelfile.jarvis-lab (Ollama config)")
    print()
    print("🚀 Next steps:")
    print("  1. Test with Ollama: ollama create jarvis-lab -f Modelfile.jarvis-lab")
    print("  2. Chat: ollama run jarvis-lab")
    print("  3. Or use the GGUF file directly with llama.cpp")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
