#!/usr/bin/env python3
"""
Fine-tune a small LLM on Ben Lab quantum experiment data using LoRA.

This script fine-tunes a base model (e.g., Llama-3.2-1B) on instruction-style
training data generated from Jarvis Lab API experiments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

DATA_PATH = "data/lab_instructions.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Change if needed
OUTPUT_DIR = "ben-lab-lora"


def load_lab_dataset(path: str):
    """Load JSONL data and convert to HuggingFace Dataset format."""
    try:
        from datasets import Dataset
    except ImportError:
        print("❌ Error: 'datasets' package not found")
        print("Install with: pip install datasets")
        raise

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            out = ex.get("output", "")
            if inp:
                prompt = f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nAnswer:\n"
            else:
                prompt = f"Instruction:\n{instruction}\n\nAnswer:\n"
            records.append({"prompt": prompt, "response": out})
    return Dataset.from_list(records)


@dataclass
class FormatConfig:
    """Configuration for instruction format."""
    add_system: bool = True
    system_prompt: str = "You are Ben's lab assistant, expert in quantum phases and Jarvis-2v experiments."


def make_tokenize_fn(tokenizer, cfg: FormatConfig):
    """Create tokenization function for the dataset."""
    def tokenize(batch):
        texts = []
        for p, r in zip(batch["prompt"], batch["response"]):
            if cfg.add_system:
                full = f"<s>[SYSTEM] {cfg.system_prompt}\n[/SYSTEM]\n[USER]\n{p}[/USER]\n[ASSISTANT]\n{r}[/ASSISTANT]</s>"
            else:
                full = f"{p}{r}"
            texts.append(full)
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=1024,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize


def main() -> None:
    print("🔧 Ben Lab LLM Fine-tuning with LoRA")
    print("=" * 60)
    print()

    # Check dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"❌ Error: Missing required package - {e}")
        print()
        print("Install dependencies with:")
        print("  pip install transformers>=4.40 datasets accelerate peft torch")
        return

    # Load dataset
    print(f"📂 Loading dataset from {DATA_PATH}...")
    if not Path(DATA_PATH).exists():
        print(f"❌ Error: {DATA_PATH} not found")
        print("Run generate_lab_training_data_live.py first to create training data")
        return

    raw_ds = load_lab_dataset(DATA_PATH)
    ds = raw_ds.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    print(f"✓ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print()

    # Load model and tokenizer
    print(f"🤖 Loading base model: {BASE_MODEL}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Try a different model:")
        print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("  - microsoft/phi-2")
        print("  - Qwen/Qwen2.5-1.5B-Instruct")
        return

    print()

    # Apply LoRA
    print("🔗 Applying LoRA (Low-Rank Adaptation)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    tokenize_fn = make_tokenize_fn(tokenizer, FormatConfig())

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)
    print("✓ Tokenization complete")
    print()

    # Training configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,          # Increase for more training
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    def data_collator(features):
        """Collate batch for training."""
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch

    # Create trainer
    print("🏋️  Starting training...")
    print(f"📊 Epochs: {training_args.num_train_epochs}")
    print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
    print(f"📊 Learning rate: {training_args.learning_rate}")
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter
    print()
    print(f"💾 Saving LoRA adapter to {OUTPUT_DIR}/...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✓ Training complete!")
    print()
    print("Next step: Convert LoRA adapter to GGUF and install in Ollama")
    print("Run: ./train_and_install.sh (or follow manual steps)")


if __name__ == "__main__":
    main()
