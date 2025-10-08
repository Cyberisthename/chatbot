import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "distilgpt2"  # Using a smaller model that will fit in memory
DATA_PATH = "./training-data"            # Folder with .txt or .jsonl files
OUTPUT_DIR = "./jarvis-model"
BATCH_SIZE = 2
EPOCHS = 3
MAX_LENGTH = 1024
LICENSE = """
Custom J.A.R.V.I.S. Model
Created by: Ben
License: Proprietary
All rights reserved.
This is a custom-trained model, and all rights are reserved by the creator.
"""

# Save license
with open(os.path.join(OUTPUT_DIR, "LICENSE"), "w") as f:
    f.write(LICENSE)

# --- Load Dataset ---
def load_local_dataset(data_path):
    # Assumes .txt files, one document per line
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError("No .txt files found in training-data/")
    dataset = load_dataset("text", data_files=files)
    return dataset

dataset = load_local_dataset(DATA_PATH)

# --- Tokenizer & Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# --- Tokenize ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- Data Collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,  # Reduced batch size
    gradient_accumulation_steps=4,   # Accumulate gradients
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    prediction_loss_only=True,
    fp16=False,  # Disabled FP16 for CPU training
    no_cuda=True,  # Force CPU training
    report_to="none"
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
)

# --- Train ---
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")
