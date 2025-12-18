#!/usr/bin/env python3
"""
Train a model for Ollama with HuggingFace books or local data.
This script will train on institutional books data if available, or use fallback data.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset

print("=" * 80)
print("🤖 Training JARVIS for Ollama")
print("=" * 80)

# Configuration
MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./jarvis-model"
EPOCHS = 3
BATCH_SIZE = 2
MAX_LENGTH = 512
TRAIN_SIZE = 5000

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
Path("./gguf-exports").mkdir(exist_ok=True, parents=True)

print(f"\n🔍 System Info:")
print(f"  PyTorch: {torch.__version__}")
print(f"  GPU Available: {torch.cuda.is_available()}")

# Step 1: Prepare training data
print(f"\n📚 Preparing training data...")

# Try to load the institutional books dataset (requires HF token)
dataset = None
try:
    print(f"  Attempting to load institutional-books-1.0 dataset...")
    # This requires authentication via huggingface login
    from huggingface_hub import login
    # Uncomment to login if you have token
    # login("your_hf_token_here")
    
    dataset = load_dataset("institutional/institutional-books-1.0")
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset[list(dataset.keys())[0]]
    
    print(f"  ✅ Dataset loaded: {len(train_dataset)} samples")
    
    if len(train_dataset) > TRAIN_SIZE:
        print(f"  Limiting to {TRAIN_SIZE} samples")
        train_dataset = train_dataset.select(range(TRAIN_SIZE))
        
except Exception as e:
    print(f"  ⚠️ Could not load HuggingFace dataset: {e}")
    print(f"  Using fallback training data instead")
    
    # Fallback: Create training data about books, knowledge, and AI
    training_texts = [
        # Academic and knowledge content
        "The history of artificial intelligence spans over seven decades of research and development. From the Turing test to modern deep learning, AI has evolved tremendously.",
        "Machine learning enables computers to learn from data without being explicitly programmed. Supervised learning, unsupervised learning, and reinforcement learning are three main paradigms.",
        "Natural language processing is a subdiscipline of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Deep neural networks with multiple layers can learn complex patterns in data. Convolutional neural networks excel at image processing, while recurrent neural networks handle sequences.",
        "Transformers revolutionized machine learning with attention mechanisms, enabling models like BERT, GPT, and T5 to achieve state-of-the-art results across many NLP tasks.",
        
        # Books and literature content
        "Books are a gateway to worlds of imagination and knowledge. From fiction to non-fiction, they provide entertainment, education, and inspiration.",
        "The invention of the printing press in 1440 revolutionized the distribution of knowledge, making books more accessible to the general population.",
        "Libraries serve as repositories of human knowledge, preserving books and providing access to information for generations.",
        "Digital libraries and e-books have transformed how we access and read literature in the modern era.",
        "Reading develops critical thinking skills, expands vocabulary, and provides opportunities to explore different perspectives and ideas.",
        
        # Professional and technical content
        "Software engineering best practices include version control, testing, documentation, and continuous integration.",
        "Cloud computing has transformed infrastructure management, allowing organizations to scale resources on demand.",
        "Data science combines statistics, programming, and domain knowledge to extract insights from data.",
        "DevOps practices streamline the development and deployment process, improving collaboration between teams.",
        "Cybersecurity is critical for protecting systems and data from unauthorized access and attacks.",
        
        # Educational content
        "Education is the foundation of human development and social progress. It empowers individuals and communities.",
        "Online learning platforms have democratized education, making quality courses accessible to people worldwide.",
        "Critical thinking and problem-solving skills are essential for success in the modern world.",
        "Continuous learning and adaptation are key to thriving in rapidly changing industries.",
        "Mentorship and collaboration foster knowledge sharing and accelerate professional growth.",
        
        # AI and future content
        "Artificial intelligence will likely be one of the most important innovations of the 21st century.",
        "Ethical AI development requires consideration of bias, fairness, transparency, and accountability.",
        "The future of work will involve collaboration between humans and AI systems, each complementing the other's strengths.",
        "Interdisciplinary research combining AI with fields like neuroscience, psychology, and sociology yields valuable insights.",
        "Responsible AI development requires input from diverse stakeholders including researchers, policymakers, and the public.",
    ]
    
    # Expand training data by repeating and varying
    expanded_texts = training_texts * 3
    
    train_dataset = Dataset.from_dict({"text": expanded_texts})
    print(f"  ✅ Created fallback dataset: {len(train_dataset)} samples")

# Step 2: Load model and tokenizer
print(f"\n🚀 Loading model: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    print(f"  ✅ Model loaded")
    print(f"  Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"  ❌ Error loading model: {e}")
    sys.exit(1)

# Step 3: Tokenize data
print(f"\n🔄 Tokenizing data...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

try:
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    print(f"  ✅ Tokenized {len(tokenized_dataset)} samples")
except Exception as e:
    print(f"  ⚠️ Tokenization warning: {e}")

# Step 4: Setup training
print(f"\n⚡ Setting up training...")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    use_cpu=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 5: Train
print(f"\n🎯 Starting training...")
try:
    trainer.train()
    print(f"  ✅ Training complete!")
except KeyboardInterrupt:
    print(f"  ⚠️ Training interrupted")
except Exception as e:
    print(f"  ❌ Training error: {e}")
    sys.exit(1)

# Step 6: Save model
print(f"\n💾 Saving model...")
try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Model saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Save metadata
metadata = {
    "name": "J.A.R.V.I.S. Ollama Model",
    "creator": "Ben",
    "version": "1.0.0",
    "base_model": MODEL_NAME,
    "description": "Custom-trained J.A.R.V.I.S. model for Ollama",
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Step 7: Create Ollama Modelfile
print(f"\n📝 Creating Ollama configuration...")

modelfile = f"""FROM {os.path.abspath(OUTPUT_DIR)}

TEMPLATE \"\"\"[INST] {{{{ .System }}}} {{{{ .Prompt }}}} [/INST]\"\"\"

SYSTEM \"\"\"You are J.A.R.V.I.S., an advanced AI assistant created by Ben.
You are helpful, knowledgeable, and always strive to provide accurate information.
You can assist with various tasks including writing, analysis, coding, and creative work.\"\"\"

PARAMETER num_ctx 512
PARAMETER num_predict 256
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
"""

modelfile_path = os.path.join("./gguf-exports", "Modelfile")
with open(modelfile_path, "w") as f:
    f.write(modelfile)

print(f"  ✅ Modelfile created")

# Step 8: Create integration files
print(f"\n🔧 Creating integration files...")

# Python integration module
integration_code = '''"""J.A.R.V.I.S. Ollama Integration"""
import requests
import json
from typing import Optional

class OllamaJarvis:
    def __init__(self, model="jarvis", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
    
    def chat(self, prompt: str, stream: bool = False) -> str:
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": stream},
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

if __name__ == "__main__":
    jarvis = OllamaJarvis()
    
    print("🤖 J.A.R.V.I.S. - Ollama Chat")
    print("Type 'quit' to exit\\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            response = jarvis.chat(user_input)
            print(f"J.A.R.V.I.S.: {response}\\n")
        except Exception as e:
            print(f"Error: {e}\\n")
'''

integration_path = os.path.join("./gguf-exports", "ollama_jarvis.py")
with open(integration_path, "w") as f:
    f.write(integration_code)

# Setup instructions
setup_guide = """# J.A.R.V.I.S. + Ollama Setup Guide

## Prerequisites
1. Install Ollama: https://ollama.ai
2. Ensure Python 3.8+ is available
3. Install requests: `pip install requests`

## Setup Steps

### 1. Download Ollama
- Visit https://ollama.ai
- Download the appropriate version for your OS (Windows, macOS, Linux)
- Install and run Ollama

### 2. Verify Ollama is Running
```bash
curl http://localhost:11434/api/tags
```

### 3. Create the J.A.R.V.I.S. Model
```bash
cd ./gguf-exports
ollama create jarvis -f ./Modelfile
```

### 4. Test the Model
```bash
ollama run jarvis

# In the prompt, try:
# > Who are you?
# > Tell me about artificial intelligence
# > Help me with Python coding
```

### 5. Use via Python API
```bash
python3 ollama_jarvis.py
```

### 6. Use via cURL
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "jarvis",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

## Integration with Node.js

Update your `server.js` or inference API to use Ollama:

```javascript
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  body: JSON.stringify({
    model: 'jarvis',
    prompt: userMessage,
    stream: false
  })
});
const data = await response.json();
return data.response;
```

## Performance Tips

- GPU: Ollama automatically uses GPU if available
- Temperature: Lower (0.3) for factual, Higher (0.9) for creative
- Context Size: Can be increased in Modelfile (num_ctx)
- Generation Length: Adjust num_predict parameter

## Troubleshooting

- **Connection refused**: Start Ollama with `ollama serve`
- **Model not found**: Verify creation with `ollama list`
- **Out of memory**: Reduce context size or batch size
- **Slow generation**: Check if GPU is being used

## Advanced Usage

### Customize Model Parameters
Edit `./Modelfile` and recreate:
```bash
ollama create jarvis -f ./Modelfile
```

### Use Different Base Models
Change the FROM line in Modelfile to other Ollama models:
- `FROM mistral` - Mistral 7B
- `FROM neural-chat` - Neural Chat
- `FROM dolphin-mixtral` - Dolphin Mixtral

### Export Model
```bash
ollama pull jarvis
# Model saved to ~/.ollama/models
```

## Next Steps

1. Fine-tune model with custom data
2. Create different model variants for different tasks
3. Deploy as API service
4. Integrate with web applications
5. Add RAG (Retrieval Augmented Generation) capabilities

---
For more information, visit: https://ollama.ai
"""

setup_path = os.path.join("./gguf-exports", "SETUP.md")
with open(setup_path, "w") as f:
    f.write(setup_guide)

print(f"  ✅ Integration files created")

# Final summary
print("\n" + "=" * 80)
print("✅ TRAINING AND SETUP COMPLETE!")
print("=" * 80)

print(f"\n📁 Trained Model: {os.path.abspath(OUTPUT_DIR)}")
print(f"📁 Ollama Files: {os.path.abspath('./gguf-exports')}")

print(f"\n🚀 Quick Start:")
print(f"   1. Install Ollama: https://ollama.ai")
print(f"   2. Start Ollama: ollama serve")
print(f"   3. Create model: ollama create jarvis -f ./gguf-exports/Modelfile")
print(f"   4. Chat: ollama run jarvis")

print(f"\n📚 Documentation: ./gguf-exports/SETUP.md")
print(f"🐍 Python integration: ./gguf-exports/ollama_jarvis.py")

print("=" * 80)
