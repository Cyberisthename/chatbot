# JARVIS-2v Training Quick Start

Get started training JARVIS-2v adapters in 5 minutes.

## ⚠️ Critical Understanding

**JARVIS-2v's intelligence is NOT in the base model.**

Your AI's intelligence lives in:
- **AdapterEngine** - Modular knowledge units
- **Y/Z/X Routing** - Bit-based task selection
- **Memory System** - Persistent facts and context
- **Quantum Artifacts** - Experimental data

**The GGUF model is ONLY a language decoder.**

Training = Creating adapters + Building memory (NOT fine-tuning)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies for training:
- `datasets` - HuggingFace datasets for IDI streaming
- `networkx` - Adapter graph relationships
- `pyyaml` - Configuration management

### 2. Train from Local Files

```bash
# Add your knowledge to data/raw/
echo "JARVIS-2v uses adapter-based intelligence" > data/raw/knowledge.txt

# Train
python scripts/train_adapters.py --input data/raw

# Output:
#   📄 Processing: knowledge.txt
#       Split into 1 chunks
#       Detected domains: general
#       ✅ Created 1 adapters
```

### 3. Stream from IDI Dataset (Optional)

```bash
# Stream 50 books from Institutional Data Initiative
python scripts/train_idi_stream.py --max-books 50 --language en

# This will:
# - Stream books incrementally (no full download)
# - Create adapters per chunk
# - Track progress in idi_training_metadata.json
# - Save every 10 books
```

### 4. Quick Knowledge Ingestion

```bash
# Add single fact
python scripts/ingest_knowledge.py --fact "Ben is JARVIS's creator"

# Add from file
python scripts/ingest_knowledge.py --file data/facts.txt --tags general

# Show stats
python scripts/ingest_knowledge.py --stats
```

### 5. Verify Training

```bash
# Check adapter count
ls adapters/*.json | wc -l

# View adapter details
cat adapters/adapter_*.json | jq .

# Check memory
cat jarvis_memory.json | jq .facts
```

### 6. Test the System

```bash
# Start backend
python inference.py models/jarvis-7b-q4_0.gguf --port 8000 &

# Test query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What do you know?"}]}'
```

## 📚 Training Methods Comparison

| Method | Use Case | Speed | Data Source |
|--------|----------|-------|-------------|
| **train_adapters.py** | Local files | Fast | Your files |
| **train_idi_stream.py** | Large datasets | Streaming | HuggingFace |
| **ingest_knowledge.py** | Quick facts | Instant | Manual input |

## 🎯 Common Use Cases

### Case 1: Add Domain Knowledge

```bash
# Create domain files
mkdir -p data/raw/quantum
cat > data/raw/quantum/basics.txt << EOF
Qubits are quantum bits that can be in superposition.
Entanglement links quantum states across space.
Quantum gates manipulate qubit states.
EOF

# Train
python scripts/train_adapters.py --input data/raw/quantum

# Result: Adapters tagged with "quantum" domain
```

### Case 2: Build Programming Knowledge

```bash
# Add code examples
mkdir -p data/raw/code
cat > data/raw/code/python_basics.md << EOF
# Python Basics

Functions are defined with the def keyword.
Classes use __init__ for initialization.
List comprehensions: [x*2 for x in range(10)]
EOF

# Train with specific extensions
python scripts/train_adapters.py --input data/raw/code --extensions .md .py

# Result: Programming adapters with Y-bit 0 set
```

### Case 3: Continuous Learning Loop

```bash
#!/bin/bash
# continuous_training.sh

while true; do
    # Check for new files
    python scripts/train_adapters.py --input data/raw/daily
    
    # Ingest from knowledge queue
    if [ -f data/knowledge_queue.txt ]; then
        python scripts/ingest_knowledge.py --file data/knowledge_queue.txt
        rm data/knowledge_queue.txt
    fi
    
    # Wait 1 hour
    sleep 3600
done
```

## 🔍 Monitoring Progress

### View Training Stats

```bash
# Adapter count
echo "Adapters: $(ls adapters/*.json 2>/dev/null | wc -l)"

# Memory facts
echo "Facts: $(cat jarvis_memory.json | jq '.facts | length')"

# IDI progress
cat idi_training_metadata.json | jq '{books: .books_processed | length, adapters: .total_adapters}'

# Local progress
cat local_training_metadata.json | jq '{files: .files_processed | length, adapters: .total_adapters}'
```

### Inspect Adapter Graph

```bash
# View graph structure
cat adapters_graph.json | jq .

# Count relationships
cat adapters_graph.json | jq '.links | length'

# Find adapter by tag
find adapters -name "*.json" -exec grep -l "quantum" {} \;
```

## ⚙️ Configuration

### Adjust Chunk Size

```bash
# Smaller chunks = more granular (256 words)
python scripts/train_adapters.py --input data/raw --chunk-size 256

# Larger chunks = more context (1024 words)
python scripts/train_adapters.py --input data/raw --chunk-size 1024
```

### Filter IDI by Language

```bash
# English only (default)
python scripts/train_idi_stream.py --max-books 100 --language en

# French
python scripts/train_idi_stream.py --max-books 100 --language fr

# Any language (no filter)
python scripts/train_idi_stream.py --max-books 100 --language ""
```

### Custom Config File

```bash
# Use custom config
python scripts/train_adapters.py --config ./my_config.yaml
```

## 🛠️ Troubleshooting

### No adapters created

**Symptom**: Script runs but no adapters appear

**Solutions**:
```bash
# Check input directory exists
ls -lh data/raw/

# Check for compatible files
ls data/raw/*.{txt,md,json,csv}

# Verify permissions
chmod +r data/raw/*
```

### IDI streaming fails

**Symptom**: "datasets library not found" or connection error

**Solutions**:
```bash
# Install datasets
pip install datasets

# Check internet connection
curl -I https://huggingface.co

# Try with fewer books
python scripts/train_idi_stream.py --max-books 10
```

### Memory not persisting

**Symptom**: Facts disappear after restart

**Solutions**:
```bash
# Check file permissions
ls -lh jarvis_memory.json

# Verify file is being written
cat jarvis_memory.json | jq .

# Manual save test
python scripts/ingest_knowledge.py --fact "test" --stats
```

## 📈 Next Steps

1. **Read Full Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)
2. **Deploy Your System**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
3. **Set Up Ollama**: [ollama/README.md](ollama/README.md)
4. **Explore Quantum Lab**: Generate synthetic experiments

## 💡 Pro Tips

1. **Start Small**: Train on 10-20 documents first, then scale up
2. **Organize by Domain**: Keep knowledge organized in subdirectories
3. **Backup Regularly**: `tar -czf adapters-backup.tar.gz adapters/`
4. **Monitor Memory Usage**: Keep facts under 1000 for optimal performance
5. **Test After Training**: Always verify adapters work with test queries

## 🎓 Understanding Output

When you run training, you'll see:

```
📁 Found 5 files to process

  📄 Processing: knowledge.txt
      Split into 3 chunks
      Detected domains: general, ai
      ✅ Created 3 adapters

  📄 Processing: quantum_basics.md
      Split into 8 chunks
      Detected domains: science, quantum
      ✅ Created 8 adapters

💾 Progress saved: 5 files, 25 adapters

✅ Local Training Complete!
   Files processed: 5
   Adapters created: 25
   Metadata saved to: ./local_training_metadata.json
```

This means:
- ✅ 5 files were processed successfully
- ✅ 25 new adapters were created (non-destructive)
- ✅ Each adapter has domain tags and Y-bits
- ✅ Adapters are linked in sequence (chunks)
- ✅ Memory was updated with source information
- ✅ Progress is tracked for resumability

## 🚀 Ready to Go!

You're now ready to train JARVIS-2v. Remember:

- Training = Creating adapters (NOT fine-tuning)
- All learning is non-destructive and incremental
- Your intelligence lives in adapters, not model weights
- The base model is just a language decoder

**Happy Training!** 🎉

---

For detailed information, see:
- [Full Training Guide](docs/TRAINING.md)
- [Architecture Overview](README.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
