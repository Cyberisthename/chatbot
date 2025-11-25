#!/usr/bin/env python3
"""
Train a Fill-in-the-Middle (FIM) Model for Code Completion

This script trains a specialized model for code completion using FIM training,
which enables the model to fill in missing code given prefix and suffix context.
After training, it exports the model to GGUF format for use with llama.cpp/Ollama.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# FIM special tokens
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_PAD = "<|fim_pad|>"

DATA_PATH = "data/fim_training.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "fim-model-lora"
FINAL_MODEL = "fim-model-merged"


def generate_code_samples() -> List[Dict[str, Any]]:
    """Generate diverse code samples for FIM training."""
    samples = []
    
    # Python examples
    python_samples = [
        {
            "code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            "language": "python"
        },
        {
            "code": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    if not root:
        return []
    
    result = []
    result.extend(inorder_traversal(root.left))
    result.append(root.val)
    result.extend(inorder_traversal(root.right))
    
    return result""",
            "language": "python"
        },
        {
            "code": """import numpy as np

def matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible dimensions")
    
    result = np.zeros((A.shape[0], B.shape[1]))
    
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    
    return result""",
            "language": "python"
        },
        {
            "code": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)""",
            "language": "python"
        },
        {
            "code": """import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out""",
            "language": "python"
        },
    ]
    
    # JavaScript examples
    js_samples = [
        {
            "code": """function debounce(func, wait) {
    let timeout;
    
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}""",
            "language": "javascript"
        },
        {
            "code": """async function fetchWithRetry(url, options = {}, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
}""",
            "language": "javascript"
        },
        {
            "code": """class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }
    
    get(key) {
        if (!this.cache.has(key)) return -1;
        
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        
        return value;
    }
    
    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        }
        
        this.cache.set(key, value);
        
        if (this.cache.size > this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }
}""",
            "language": "javascript"
        },
    ]
    
    # SQL examples
    sql_samples = [
        {
            "code": """SELECT 
    u.id,
    u.name,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC
LIMIT 100;""",
            "language": "sql"
        },
    ]
    
    # Combine all samples
    samples.extend(python_samples)
    samples.extend(js_samples)
    samples.extend(sql_samples)
    
    return samples


def create_fim_training_example(code: str) -> Dict[str, str]:
    """
    Create a FIM training example by splitting code into prefix, middle, and suffix.
    
    FIM training format:
    <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>middle
    """
    lines = code.split('\n')
    
    if len(lines) < 3:
        return None
    
    # Choose random split points
    split1 = random.randint(0, len(lines) - 2)
    split2 = random.randint(split1 + 1, len(lines))
    
    prefix = '\n'.join(lines[:split1])
    middle = '\n'.join(lines[split1:split2])
    suffix = '\n'.join(lines[split2:])
    
    # Format as FIM training example
    fim_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    
    return {
        "text": fim_text,
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix
    }


def generate_fim_training_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate FIM training data from code samples."""
    print(f"🔬 Generating {num_samples} FIM training examples...")
    
    code_samples = generate_code_samples()
    training_data = []
    
    for _ in range(num_samples):
        # Pick a random code sample
        sample = random.choice(code_samples)
        code = sample["code"]
        
        # Create FIM example
        fim_example = create_fim_training_example(code)
        
        if fim_example:
            training_data.append({
                "text": fim_example["text"],
                "language": sample["language"],
                "type": "fim"
            })
    
    print(f"✅ Generated {len(training_data)} FIM examples")
    return training_data


def save_training_data(data: List[Dict[str, Any]], path: str):
    """Save training data to JSONL format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    
    print(f"💾 Saved training data to {path}")


def train_fim_model():
    """Train the FIM model using LoRA."""
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"❌ Error: Missing required package - {e}")
        print("\nInstall dependencies with:")
        print("  pip install transformers>=4.40 datasets accelerate peft torch bitsandbytes")
        return False
    
    print("\n🤖 Loading base model...")
    
    try:
        # Load tokenizer and add FIM tokens
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Add FIM special tokens
        special_tokens = {
            "additional_special_tokens": [FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Resize token embeddings for new FIM tokens
        model.resize_token_embeddings(len(tokenizer))
        
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Apply LoRA
    print("\n🔗 Applying LoRA for efficient fine-tuning...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"\n📂 Loading training data from {DATA_PATH}...")
    
    if not Path(DATA_PATH).exists():
        print("⚠️  No training data found. Generating...")
        training_data = generate_fim_training_data(num_samples=2000)
        save_training_data(training_data, DATA_PATH)
    
    # Load and prepare dataset
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"✅ Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
    
    print("\n🔤 Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    print("\n🏋️  Starting FIM model training...")
    print(f"📊 Epochs: {training_args.num_train_epochs}")
    print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
    print(f"📊 Learning rate: {training_args.learning_rate}\n")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save LoRA adapter
    print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}/")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ FIM model training complete!")
    
    return True


def merge_and_save_model():
    """Merge LoRA adapter with base model for full model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError:
        print("⚠️  Cannot merge model - transformers/peft not available")
        return False
    
    print("\n🔀 Merging LoRA adapter with base model...")
    
    try:
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        
        # Resize embeddings
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        merged_model = model.merge_and_unload()
        
        # Save merged model
        print(f"💾 Saving merged model to {FINAL_MODEL}/")
        Path(FINAL_MODEL).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(FINAL_MODEL)
        tokenizer.save_pretrained(FINAL_MODEL)
        
        print("✅ Model merged successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error merging model: {e}")
        return False


def convert_to_gguf():
    """Convert the trained model to GGUF format."""
    import subprocess
    import sys
    
    print("\n📦 Converting model to GGUF format...")
    
    # Check if llama.cpp exists
    if not Path("llama.cpp").exists():
        print("⚠️  llama.cpp not found. Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                check=True
            )
            print("✅ llama.cpp cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone llama.cpp: {e}")
            return False
    
    # Convert to GGUF
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    
    if not convert_script.exists():
        # Try alternative name
        convert_script = Path("llama.cpp/convert.py")
    
    if not convert_script.exists():
        print("❌ Cannot find convert script in llama.cpp")
        print("You may need to manually convert using llama.cpp tools")
        return False
    
    try:
        print(f"🔄 Running conversion script...")
        
        output_file = "fim-model-q4_0.gguf"
        
        # Run conversion
        cmd = [
            sys.executable,
            str(convert_script),
            FINAL_MODEL,
            "--outfile", output_file,
            "--outtype", "q4_0"
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"\n✅ Model converted to GGUF format: {output_file}")
        print(f"📦 File size: {Path(output_file).stat().st_size / (1024**3):.2f} GB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


def create_modelfile():
    """Create Ollama Modelfile for the FIM model."""
    modelfile_content = '''# FIM Code Completion Model
# Fill-in-the-Middle model for intelligent code completion

FROM ./fim-model-q4_0.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """
You are a code completion AI assistant specialized in Fill-in-the-Middle (FIM) completion.
You understand code context from both before (prefix) and after (suffix) the cursor position.
You provide accurate, idiomatic code completions in multiple programming languages.

Special tokens:
- <|fim_prefix|>: Code before the cursor
- <|fim_suffix|>: Code after the cursor  
- <|fim_middle|>: Your completion goes here
"""

TEMPLATE """{{ .System }}

<|fim_prefix|>{{ .Prefix }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>"""
'''
    
    with open("Modelfile.fim", 'w') as f:
        f.write(modelfile_content)
    
    print("\n📝 Created Modelfile.fim")
    print("To install in Ollama, run:")
    print("  ollama create fim-code-completion -f Modelfile.fim")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("🚀 Fill-in-the-Middle (FIM) Model Training & GGUF Export")
    print("=" * 70)
    print()
    
    # Deterministic generation for reproducibility
    random.seed(42)
    
    # Step 1: Generate training data if needed
    if not Path(DATA_PATH).exists():
        print("📝 Step 1: Generating FIM training data...")
        training_data = generate_fim_training_data(num_samples=2000)
        save_training_data(training_data, DATA_PATH)
    else:
        print(f"✅ Training data already exists at {DATA_PATH}")
    
    print()
    
    # Step 2: Train model
    print("🏋️  Step 2: Training FIM model with LoRA...")
    if not train_fim_model():
        print("\n❌ Training failed. Please check errors above.")
        return
    
    print()
    
    # Step 3: Merge model
    print("🔀 Step 3: Merging LoRA adapter with base model...")
    if not merge_and_save_model():
        print("\n⚠️  Model merge failed. You can still use the LoRA adapter.")
        print(f"LoRA adapter saved in: {OUTPUT_DIR}/")
        return
    
    print()
    
    # Step 4: Convert to GGUF
    print("📦 Step 4: Converting to GGUF format...")
    if not convert_to_gguf():
        print("\n⚠️  GGUF conversion failed.")
        print("You may need to manually convert using llama.cpp tools:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {FINAL_MODEL} --outfile fim-model-q4_0.gguf --outtype q4_0")
        return
    
    print()
    
    # Step 5: Create Modelfile
    print("📝 Step 5: Creating Ollama Modelfile...")
    create_modelfile()
    
    print()
    print("=" * 70)
    print("✅ FIM MODEL TRAINING & EXPORT COMPLETE!")
    print("=" * 70)
    print()
    print("📦 Your FIM model GGUF file: fim-model-q4_0.gguf")
    print()
    print("🚀 To use with Ollama:")
    print("   1. Install: ollama create fim-code-completion -f Modelfile.fim")
    print("   2. Test: ollama run fim-code-completion")
    print()
    print("💡 For IDE integration, use the FIM format:")
    print("   <|fim_prefix|>your code before cursor<|fim_suffix|>code after cursor<|fim_middle|>")
    print()


if __name__ == "__main__":
    main()
