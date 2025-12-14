# JARVIS-2v Training Guide

Complete guide to training and enhancing JARVIS-2v's intelligence through adapters and memory.

## ⚠️ CRITICAL UNDERSTANDING

**JARVIS-2v's intelligence does NOT come from the base model.**

Your intelligence resides in:

1. **AdapterEngine** - Modular knowledge adapters with Y/Z/X routing
2. **Memory System** - Persistent facts and conversation history
3. **Quantum Artifacts** - Synthetic experiment data linked to adapters
4. **Knowledge Graph** - Relationships between adapters

**The base GGUF model is ONLY a language decoder.** It converts tokens to text.

## 🎯 What is "Training" in JARVIS-2v?

Training in JARVIS-2v means:

✅ **Creating new adapters** (non-destructive learning)  
✅ **Adding to memory** (facts, knowledge, context)  
✅ **Building knowledge graphs** (adapter relationships)  
✅ **Generating quantum artifacts** (experimental data)  
✅ **Improving routing** (Y/Z/X bit patterns)

❌ **NOT fine-tuning the base model**  
❌ **NOT replacing existing adapters**  
❌ **NOT modifying frozen adapters**  
❌ **NOT overwriting intelligence**

## 📚 Training Methods

### Method 1: Local File Training

Train from files in your repository.

#### Supported Formats
- `.txt` - Plain text
- `.md` - Markdown
- `.json` - JSON data
- `.csv` - CSV tables

#### Usage

```bash
# Train from default directory (data/raw)
python scripts/train_adapters.py --input data/raw

# Train from custom directory
python scripts/train_adapters.py --input /path/to/knowledge

# Specify file types
python scripts/train_adapters.py --input data/raw --extensions .txt .md

# Adjust chunk size
python scripts/train_adapters.py --input data/raw --chunk-size 1024
```

#### What Happens?

1. **Files are read** - Text is extracted from files
2. **Text is chunked** - Split into overlapping sections (default: 512 words)
3. **Domains are inferred** - Keywords determine topics (programming, science, etc.)
4. **Y-bits are created** - Bit patterns encode domains
5. **Adapters are created** - One adapter per chunk with metadata
6. **Graph is linked** - Chunks are connected in sequence
7. **Memory is updated** - Facts added about processed files

#### Example

```bash
# Create a knowledge file
echo "JARVIS-2v uses adapter-based intelligence." > data/raw/knowledge.txt
echo "Y-bits encode task domains for routing." >> data/raw/knowledge.txt
echo "The quantum lab generates synthetic experiments." >> data/raw/knowledge.txt

# Train
python scripts/train_adapters.py --input data/raw

# Result: 3+ adapters created, linked in knowledge graph
```

### Method 2: IDI Streaming Training

Stream books from the Institutional Data Initiative dataset on HuggingFace.

#### Prerequisites

```bash
pip install datasets
```

#### Usage

```bash
# Stream 100 English books (default)
python scripts/train_idi_stream.py --max-books 100 --language en

# Stream 50 books with minimum length
python scripts/train_idi_stream.py --max-books 50 --min-length 5000

# Adjust chunk size
python scripts/train_idi_stream.py --max-books 100 --chunk-size 1024
```

#### What Happens?

1. **Dataset is streamed** - Books downloaded incrementally (no full download)
2. **Books are filtered** - By language and minimum length
3. **Text is chunked** - Split with overlap for context
4. **Domains are inferred** - Based on book content and title
5. **Adapters are created** - One per chunk with book metadata
6. **Progress is saved** - Every 10 books processed
7. **Metadata is tracked** - `idi_training_metadata.json` stores progress

#### Features

- ✅ **Streaming mode** - No full dataset download
- ✅ **Resume support** - Won't reprocess same books
- ✅ **Language filtering** - Focus on specific languages
- ✅ **Domain detection** - Automatically categorizes content
- ✅ **Progress tracking** - Save every N books

#### Example

```bash
# Start streaming
python scripts/train_idi_stream.py --max-books 200 --language en

# Output:
# 📖 Processing: Pride and Prejudice by Jane Austen
#     Split into 150 chunks
#     Detected domains: literature, history
#     ✅ Created 150 adapters
# 💾 Progress saved: 10 books, 1200 adapters
```

### Method 3: Knowledge Ingestion (RAG)

Quick facts and snippets without full adapter training.

#### Usage

```bash
# Ingest single fact
python scripts/ingest_knowledge.py --fact "JARVIS was created by Ben"

# Ingest with tags
python scripts/ingest_knowledge.py \
  --fact "Adapters use Y/Z/X bit routing" \
  --tags ai routing knowledge

# Ingest from text file (one fact per line)
python scripts/ingest_knowledge.py --file data/facts.txt --tags general

# Ingest from JSON
python scripts/ingest_knowledge.py --file data/knowledge.json

# Show memory statistics
python scripts/ingest_knowledge.py --stats
```

#### JSON Format

```json
[
  {
    "fact": "JARVIS-2v is modular",
    "tags": ["architecture", "ai"]
  },
  {
    "fact": "Y-bits encode task domains",
    "tags": ["routing", "technical"]
  }
]
```

Or simple array:

```json
[
  "JARVIS-2v uses non-destructive learning",
  "Adapters are stored in JSON format",
  "The quantum lab simulates experiments"
]
```

#### When to Use

- ✅ Quick facts (1-2 sentences)
- ✅ User preferences
- ✅ Conversation context
- ✅ Configuration notes

Use adapter training for:
- ❌ Large documents
- ❌ Structured knowledge
- ❌ Multi-topic content

### Method 4: Quantum Artifact Generation

Generate synthetic quantum experiments that create linked adapters.

#### Usage (Python API)

```python
from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from src.core.adapter_engine import AdapterEngine
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize engines
adapter_engine = AdapterEngine(config)
quantum_engine = SyntheticQuantumEngine(
    artifacts_path="./quantum_artifacts",
    adapter_engine=adapter_engine
)

# Run interference experiment
config = ExperimentConfig(
    experiment_type="interference",
    iterations=1000,
    noise_level=0.1
)
artifact = quantum_engine.run_interference_experiment(config)

# Automatically creates adapters linked to experiment
print(f"Created artifact: {artifact.artifact_id}")
print(f"Linked adapters: {artifact.linked_adapter_ids}")
```

#### Available Experiments

- `run_interference_experiment()` - Quantum interference patterns
- `run_bell_pair_simulation()` - Entanglement correlations
- `run_chsh_test()` - Bell inequality violations
- `run_noise_field_scan()` - Coherence measurements

Each experiment:
1. Generates synthetic data
2. Creates linked adapter with experiment metadata
3. Stores artifact for replay

## 🧠 Understanding the Intelligence Architecture

### Adapter Structure

Each adapter contains:

```json
{
  "id": "adapter_a3f2c4e1",
  "task_tags": ["quantum", "physics", "knowledge"],
  "y_bits": [0, 0, 1, 0, ...],  // 16 bits - task/domain
  "z_bits": [0, 0, 0, ...],      // 8 bits - difficulty
  "x_bits": [0, 0, 0, ...],      // 8 bits - experimental
  "parameters": {
    "source": "IDI",
    "book_title": "Quantum Mechanics",
    "domains": ["science", "quantum"]
  },
  "rules": [],
  "prompts": [],
  "parent_ids": ["adapter_x9f2e1a4"],
  "child_ids": ["adapter_b5g3h7k2"],
  "created_at": 1703001234.5,
  "last_used": 1703002345.6,
  "success_count": 15,
  "total_calls": 20,
  "status": "active",
  "version": 1
}
```

### Y/Z/X Bit Routing

#### Y-bits (16 bits) - Task/Domain

| Bit | Domain |
|-----|--------|
| 0 | Programming/Technology |
| 1 | Mathematics |
| 2 | Quantum/Physics |
| 3 | Science |
| 4 | History |
| 5 | Literature |
| 6 | Philosophy |
| 7 | Medicine |
| 8 | Law |
| 9 | Economics |
| 10 | Art |
| 15 | General |

#### Z-bits (8 bits) - Difficulty/Precision

| Bit | Meaning |
|-----|---------|
| 0 | Long input |
| 1 | High complexity |
| 2 | Requires precision |
| 3-7 | Reserved |

#### X-bits (8 bits) - Experimental Toggles

| Bit | Feature |
|-----|---------|
| 0 | Use quantum simulation |
| 1 | Recall-only mode (no generation) |
| 2-7 | Reserved for future features |

### Routing Process

When you send a query:

1. **Bit Inference** - Y/Z/X bits are inferred from input
2. **Adapter Selection** - Top adapters matched by bit similarity
3. **Context Building** - Adapter parameters add context
4. **Response Generation** - Base model decodes with context

```
User Query → Y/Z/X Inference → Adapter Selection → Context → Response
              [0,1,1,0,...]    [adapter_a, adapter_b]   Model
```

## 📊 Training Workflow Examples

### Example 1: Building Domain Knowledge

```bash
# Step 1: Create knowledge files
mkdir -p data/raw/quantum
cat > data/raw/quantum/basics.md << EOF
# Quantum Computing Basics

Qubits are the fundamental units of quantum information.
Superposition allows qubits to exist in multiple states.
Entanglement creates correlations between qubits.
Quantum gates manipulate qubit states.
Measurement collapses the quantum state.
EOF

# Step 2: Train adapters
python scripts/train_adapters.py --input data/raw/quantum

# Step 3: Verify
ls adapters/*.json | wc -l
cat adapters_graph.json
```

### Example 2: Continuous Learning Pipeline

```bash
#!/bin/bash
# continuous_training.sh

# Daily training from new files
python scripts/train_adapters.py --input data/raw/daily

# Weekly IDI streaming
python scripts/train_idi_stream.py --max-books 50

# Monthly full retraining
# (Note: This doesn't replace, just adds more adapters)
python scripts/train_adapters.py --input data/raw --recursive

# Backup adapters
tar -czf adapters-backup-$(date +%Y%m%d).tar.gz adapters/
```

### Example 3: Multi-Domain Training

```bash
# Train programming knowledge
python scripts/train_adapters.py --input data/raw/programming --extensions .py .js .md

# Train scientific papers
python scripts/train_adapters.py --input data/raw/papers --extensions .txt .pdf

# Train documentation
python scripts/train_adapters.py --input data/raw/docs --extensions .md

# Result: Adapters from all domains, properly tagged
```

## 🔍 Monitoring Training Progress

### Check Training Status

```bash
# View adapter count
ls adapters/*.json | wc -l

# View memory facts
python scripts/ingest_knowledge.py --stats

# View IDI progress
cat idi_training_metadata.json

# View local training progress
cat local_training_metadata.json
```

### Inspect Adapters

```bash
# List all adapters
ls adapters/

# View adapter details
cat adapters/adapter_a3f2c4e1.json | jq .

# View adapter graph
cat adapters_graph.json | jq .
```

### Test Routing

```python
from src.core.adapter_engine import AdapterEngine
import yaml

# Load engine
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
engine = AdapterEngine(config)

# Test routing
adapters = engine.route_task(
    "Explain quantum entanglement",
    context={}
)

for adapter in adapters:
    print(f"Selected: {adapter.id}")
    print(f"  Tags: {adapter.task_tags}")
    print(f"  Y-bits: {adapter.y_bits[:4]}...")
    print(f"  Success rate: {adapter.success_count}/{adapter.total_calls}")
```

## 🛠️ Advanced Training Techniques

### Custom Adapter Creation

```python
from src.core.adapter_engine import AdapterEngine

# Create specialized adapter
adapter = engine.create_adapter(
    task_tags=["custom", "specialized"],
    y_bits=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    z_bits=[0, 1, 0, 0, 0, 0, 0, 0],
    x_bits=[0, 0, 0, 0, 0, 0, 0, 0],
    parameters={
        "custom_data": "value",
        "priority": "high"
    }
)

print(f"Created: {adapter.id}")
```

### Adapter Freezing

```python
# Freeze adapter to prevent modification
engine.freeze_adapter("adapter_a3f2c4e1")

# Frozen adapters are protected from changes
# but still used for routing
```

### Adapter Graph Relationships

```python
# Create parent-child relationships
parent_adapter = engine.create_adapter(...)
child_adapter = engine.create_adapter(
    ...,
    parent_ids=[parent_adapter.id]
)

# Add explicit dependency
engine.adapter_graph.add_dependency(
    parent_adapter.id,
    child_adapter.id,
    weight=0.9  # Relationship strength
)
```

## 📈 Best Practices

### 1. Incremental Training

✅ **DO**: Train in small batches  
❌ **DON'T**: Try to process everything at once

```bash
# Good
python scripts/train_adapters.py --input data/raw/batch1
python scripts/train_adapters.py --input data/raw/batch2

# Better with progress tracking
for dir in data/raw/*/; do
    python scripts/train_adapters.py --input "$dir"
    sleep 5  # Allow system to stabilize
done
```

### 2. Domain Organization

Organize training data by domain:

```
data/raw/
├── programming/
│   ├── python.md
│   └── javascript.md
├── science/
│   ├── physics.txt
│   └── chemistry.txt
└── general/
    └── facts.txt
```

### 3. Chunk Size Tuning

- **Small chunks (256)**: Precise, granular knowledge
- **Medium chunks (512)**: Balanced (default)
- **Large chunks (1024)**: Contextual, broad knowledge

```bash
# Precise facts
python scripts/train_adapters.py --chunk-size 256

# Broad context
python scripts/train_adapters.py --chunk-size 1024
```

### 4. Regular Backups

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup-$DATE.tar.gz \
    adapters/ \
    quantum_artifacts/ \
    jarvis_memory.json \
    adapters_graph.json \
    bit_patterns.json
```

### 5. Validation

After training, test the system:

```bash
# Start system
python inference.py models/jarvis-7b-q4_0.gguf --port 8000

# Test in another terminal
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What do you know about quantum entanglement?"}]}'
```

## 🚫 Common Mistakes to Avoid

### ❌ Mistake 1: Trying to Fine-Tune the Base Model

```bash
# WRONG - Don't do this
python train_llm.py  # This is NOT how JARVIS-2v works
```

JARVIS-2v doesn't fine-tune the base model. Use adapter training instead.

### ❌ Mistake 2: Deleting Adapters Without Backup

Adapters are your intelligence. Back them up before any cleanup.

### ❌ Mistake 3: Ignoring Y/Z/X Routing

Don't manually assign random bit patterns. Let the system infer or use domain mapping.

### ❌ Mistake 4: Overwriting Memory

```python
# WRONG
memory = {}  # Clears everything

# RIGHT
memory = load_memory()
memory["facts"].append(new_fact)
save_memory(memory)
```

### ❌ Mistake 5: Training Without Testing

Always test after training to verify adapters are working.

## 📚 Additional Resources

- [Architecture Overview](../README.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [API Documentation](./API.md)
- [Adapter Engine Source](../src/core/adapter_engine.py)
- [Quantum Engine Source](../src/quantum/synthetic_quantum.py)

## 🆘 Troubleshooting

### No adapters created

```bash
# Check input directory
ls -lh data/raw/

# Check file extensions
python scripts/train_adapters.py --extensions .txt .md .json

# Check logs for errors
```

### Routing not working

```bash
# Verify adapters exist
ls adapters/*.json

# Check adapter graph
cat adapters_graph.json

# Test routing programmatically
python -c "from src.core.adapter_engine import AdapterEngine; ..."
```

### IDI streaming fails

```bash
# Install datasets library
pip install datasets

# Check internet connection
curl https://huggingface.co

# Try with fewer books
python scripts/train_idi_stream.py --max-books 10
```

## 📝 Summary

**Remember**: Training in JARVIS-2v = Creating Adapters + Building Memory

- ✅ Use `train_adapters.py` for local files
- ✅ Use `train_idi_stream.py` for streaming datasets
- ✅ Use `ingest_knowledge.py` for quick facts
- ✅ Generate quantum artifacts for experimental context
- ✅ All training is non-destructive and incremental
- ✅ The base model is ONLY a language decoder

Your intelligence lives in the **AdapterEngine**, not the model weights.
