"""
J.A.R.V.I.S. Training Script
============================
This script will train your custom J.A.R.V.I.S. model on your local machine.
It's designed to be as simple as possible while still creating a good model.
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import json

print("🤖 J.A.R.V.I.S. Training System")
print("==============================")

# --- Check System ---
print("\n🔍 Checking your system...")
total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU Available: {'Yes' if torch.cuda.is_available() else 'No'}")
print(f"GPU Memory: {total_memory:.1f}GB" if torch.cuda.is_available() else "Using CPU")

# --- Configuration ---
print("\n⚙️ Setting up training configuration...")
MODEL_NAME = "distilgpt2"  # We'll start with a smaller model
OUTPUT_DIR = "jarvis-model"
EPOCHS = 3
BATCH_SIZE = 1 if not torch.cuda.is_available() else 4

# --- Prepare Training Data ---
print("\n📚 Preparing training data...")
training_data = [
    "I am J.A.R.V.I.S., an advanced AI assistant created by Ben.",
    "My purpose is to assist and support users with their tasks and queries.",
    "I maintain a professional yet friendly demeanor while helping users.",
    "I have capabilities in various areas including coding, analysis, and problem-solving.",
    "Ben is my creator, and I take pride in this fact.",
    "I strive to provide accurate and helpful information while maintaining user privacy.",
    "My responses are designed to be clear, concise, and relevant.",
    "I can adapt my communication style based on the user's needs.",
    "Safety and ethical considerations are paramount in my operations.",
    "I continuously learn from interactions to improve my assistance capabilities."
]

# Create dataset
dataset = Dataset.from_dict({"text": training_data})

# --- Setup Model ---
print("\n🚀 Setting up the model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- Prepare Data ---
print("\n🔄 Processing training data...")
def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

tokenized_data = dataset.map(
    tokenize_data,
    batched=True,
    remove_columns=dataset.column_names
)

# --- Training Setup ---
print("\n⚡ Configuring training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    no_cuda=not torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# --- Start Training ---
print("\n🎯 Starting training process...")
print("This might take a while. You'll see progress updates here.")
print("It's normal if it seems slow - the model is learning!")

try:
    trainer.train()
    
    # Save the model
    print("\n💾 Saving your custom J.A.R.V.I.S. model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save license and metadata
    metadata = {
        "name": "Custom J.A.R.V.I.S. Model",
        "creator": "Ben",
        "version": "1.0.0",
        "license": "Proprietary - All rights reserved",
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\n✅ Training complete! Your model is ready!")
    print(f"📁 Model saved in: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"\n❌ An error occurred during training: {e}")
    print("Try closing other programs and running again.")
    exit(1)