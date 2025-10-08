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
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change to your model or local path
DATA_PATH = "./training-data"            # Folder with .txt or .jsonl files
OUTPUT_DIR = "./llm-checkpoints"
BATCH_SIZE = 2
EPOCHS = 1
MAX_LENGTH = 2048

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
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    prediction_loss_only=True,
    fp16=True,  # Enable if you have a GPU with FP16 support
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
