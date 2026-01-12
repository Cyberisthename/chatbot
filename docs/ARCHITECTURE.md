# JARVIS-2v Architecture

Complete architectural overview of the JARVIS-2v modular AI system.

## 🎯 Core Philosophy

**JARVIS-2v is NOT a traditional LLM.**

Traditional LLMs:
- Intelligence in model weights
- Training = fine-tuning weights
- Fixed after training
- Black box reasoning

JARVIS-2v:
- Intelligence in adapters + memory + routing
- Training = creating new adapters
- Continuously learning
- Explainable decision paths

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      JARVIS-2v System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │   User       │────▶│  Y/Z/X Bit   │────▶│  Adapter    │  │
│  │   Query      │     │   Router     │     │  Selection  │  │
│  └──────────────┘     └──────────────┘     └─────────────┘  │
│         │                     │                     │         │
│         │                     ▼                     ▼         │
│         │              ┌──────────────┐     ┌─────────────┐  │
│         │              │   Context    │     │  Adapter    │  │
│         │              │   Builder    │◀────│   Graph     │  │
│         │              └──────────────┘     └─────────────┘  │
│         │                     │                              │
│         │                     ▼                              │
│         │              ┌──────────────┐                      │
│         └─────────────▶│   Memory     │                      │
│                        │   System     │                      │
│                        └──────────────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                      │
│                        │  Base Model  │                      │
│                        │  (Language   │                      │
│                        │   Decoder)   │                      │
│                        └──────────────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                      │
│                        │   Response   │                      │
│                        └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Core Components

### 1. AdapterEngine

**Purpose**: Modular knowledge storage and routing

**Location**: `src/core/adapter_engine.py`

**Key Classes**:
- `Adapter` - Knowledge unit with metadata
- `AdapterGraph` - NetworkX graph of relationships
- `YZXBitRouter` - Bit-based routing system
- `AdapterEngine` - Main orchestrator

**Adapter Structure**:
```json
{
  "id": "adapter_a3f2c4e1",
  "task_tags": ["quantum", "physics"],
  "y_bits": [0, 0, 1, 0, ...],  // 16 bits
  "z_bits": [0, 1, 0, ...],      // 8 bits
  "x_bits": [0, 0, 0, ...],      // 8 bits
  "parameters": {
    "source": "IDI",
    "domain": "science"
  },
  "parent_ids": ["adapter_xyz"],
  "child_ids": ["adapter_abc"],
  "success_count": 15,
  "total_calls": 20,
  "status": "active"
}
```

**Features**:
- ✅ Non-destructive learning
- ✅ Graph relationships
- ✅ Success tracking
- ✅ Freezing/deprecation
- ✅ Version control

### 2. Y/Z/X Bit Routing

**Purpose**: Intelligent adapter selection based on bit patterns

**Bit Breakdown**:

#### Y-bits (16 bits) - Task/Domain Classification
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
| 11-14 | Reserved |
| 15 | General/Unknown |

#### Z-bits (8 bits) - Difficulty/Precision
| Bit | Meaning |
|-----|---------|
| 0 | Long input |
| 1 | High complexity |
| 2 | Requires precision |
| 3 | Expert-level |
| 4-7 | Reserved |

#### X-bits (8 bits) - Experimental Toggles
| Bit | Feature |
|-----|---------|
| 0 | Use quantum simulation |
| 1 | Recall-only (no generation) |
| 2 | Verbose mode |
| 3-7 | Reserved |

**Routing Algorithm**:
1. Infer bits from user query
2. Find adapters with matching bits
3. Calculate similarity scores
4. Boost by success rate
5. Return top N adapters

**Similarity Function**:
```python
similarity = (y_match * 0.5 + z_match * 0.3 + x_match * 0.2) * success_boost
```

### 3. Memory System

**Purpose**: Persistent facts and conversation history

**Location**: `jarvis_memory.json`

**Structure**:
```json
{
  "facts": ["fact1", "fact2", ...],
  "chats": [
    {"role": "user", "content": "...", "timestamp": 123}
  ],
  "topics": {
    "quantum": 5,
    "programming": 12
  },
  "preferences": {},
  "last_topics": ["quantum", "ai", ...],
  "idi_sources": [...],
  "local_sources": [...],
  "knowledge_base": [...]
}
```

**Features**:
- ✅ Persistent storage
- ✅ Topic tracking
- ✅ Source attribution
- ✅ Thread-safe access

### 4. Synthetic Quantum Engine

**Purpose**: Generate experimental artifacts for context

**Location**: `src/quantum/synthetic_quantum.py`

**Experiments**:
- Interference patterns
- Bell pair simulations
- CHSH inequality tests
- Noise field scans

**Artifact Structure**:
```json
{
  "artifact_id": "interference_abc123",
  "experiment_type": "interference_experiment",
  "config": {...},
  "results": {
    "pattern": [...],
    "statistics": {...}
  },
  "linked_adapter_ids": ["adapter_xyz"],
  "metadata": {
    "synthetic_simulation": true
  }
}
```

**Workflow**:
1. Run experiment with config
2. Generate synthetic data
3. Create linked adapter
4. Store artifact for replay
5. Use as context for queries

### 5. Base Model (Language Decoder)

**Purpose**: Convert tokens to text (ONLY)

**Location**: Model defined in `config.yaml`

**Role**:
- ✅ Text generation
- ✅ Token prediction
- ✅ Grammar/syntax
- ❌ NOT the intelligence
- ❌ NOT trained
- ❌ NOT fine-tuned

**Integration**:
```python
# Context from adapters + memory
context = build_context(adapters, memory)

# Base model only decodes
response = model.generate(context + prompt)
```

## 🔄 Data Flow

### Request Processing

```
1. User Query
   "Explain quantum entanglement"
   
2. Y/Z/X Inference
   Y: [0,0,1,0,...]  (quantum domain)
   Z: [0,0,0,...]    (standard difficulty)
   X: [0,0,0,...]    (no special flags)
   
3. Adapter Selection
   - adapter_quantum_001 (score: 0.95)
   - adapter_physics_023 (score: 0.87)
   - adapter_science_045 (score: 0.72)
   
4. Context Building
   - Adapter parameters
   - Memory facts about quantum
   - Recent conversation context
   
5. Model Generation
   - Base model receives enriched context
   - Generates response using language patterns
   
6. Response
   "Quantum entanglement is a phenomenon where..."
   (enriched by adapter knowledge)
   
7. Feedback
   - Update adapter success_count
   - Store conversation in memory
   - Update topic tracking
```

### Training Flow

```
1. Training Data
   - Local files (data/raw/)
   - IDI streaming (HuggingFace)
   - Manual ingestion
   
2. Processing
   - Text chunking (512 words, 128 overlap)
   - Domain inference (keyword analysis)
   - Y-bit assignment (domain mapping)
   
3. Adapter Creation
   - Create adapter per chunk
   - Link to parent (sequential)
   - Store metadata (source, domain, preview)
   - Add to graph
   
4. Memory Update
   - Add facts about source
   - Track topic statistics
   - Record timestamp
   
5. Persistence
   - Save adapters to disk
   - Update graph JSON
   - Save metadata
   - Sync memory file
```

## 🔧 Configuration System

### config.yaml

```yaml
engine:
  mode: "standard"  # low_power | standard | jetson_orin
  
model:
  path: "./models/jarvis-7b-q4_0.gguf"
  gpu_layers: 0
  device: "cpu"
  
adapters:
  storage_path: "./adapters"
  graph_path: "./adapters_graph.json"
  auto_create: true
  freeze_after_creation: true
  
bits:
  y_bits: 16
  z_bits: 8
  x_bits: 8
  
memory:
  file: "./jarvis_memory.json"
  max_facts: 1000
  
quantum:
  artifacts_path: "./quantum_artifacts"
  simulation_mode: true
```

## 🚀 Performance Characteristics

### Resource Usage

| Mode | RAM | CPU | GPU | Power |
|------|-----|-----|-----|-------|
| low_power | 500MB | 10% | - | 5W |
| standard | 2GB | 30% | - | 15W |
| jetson_orin | 4GB | 20% | 50% | 25W |

### Latency

| Operation | Time |
|-----------|------|
| Bit inference | <1ms |
| Adapter selection | 5-10ms |
| Context building | 10-20ms |
| Model generation | 50-500ms |
| **Total** | **100-600ms** |

### Scaling

| Metric | Capacity |
|--------|----------|
| Adapters | 10,000+ |
| Memory facts | 1,000+ |
| Graph edges | 50,000+ |
| Concurrent users | 10+ |

## 🔍 Decision Explainability

JARVIS-2v logs every decision:

```
🔀 Routing: Y=[0,0,1,0]... Z=[0,0,0,0]... X=[0,0,0,0]... 
   -> [adapter_quantum_001, adapter_physics_023]

📊 Adapter Selection:
   - adapter_quantum_001: score=0.95, success=15/20
   - adapter_physics_023: score=0.87, success=12/15

🧠 Context Built:
   - 2 adapter parameters
   - 5 memory facts
   - 3 recent topics
```

## 🛡️ Non-Destructive Learning

Key principles:

1. **Never overwrite** - Old adapters are frozen, not deleted
2. **Always append** - New adapters added to graph
3. **Version control** - Adapters have version numbers
4. **Rollback capable** - Can revert to previous states
5. **Explainable** - Every change is logged

## 🎯 Design Decisions

### Why Adapters?

Traditional fine-tuning:
- ❌ Destroys previous knowledge (catastrophic forgetting)
- ❌ Requires full retraining
- ❌ Black box changes
- ❌ Can't undo

Adapter-based:
- ✅ Preserves all knowledge
- ✅ Incremental learning
- ✅ Explainable decisions
- ✅ Can freeze/unfreeze

### Why Y/Z/X Bits?

Alternative: Semantic embeddings
- ❌ Requires embedding model
- ❌ Slow similarity search
- ❌ Not interpretable

Bit-based:
- ✅ Fast exact matching
- ✅ Human-readable
- ✅ No extra models
- ✅ Deterministic

### Why Synthetic Quantum?

- ✅ Provides rich context data
- ✅ Demonstrates complex reasoning
- ✅ Generates linked knowledge
- ✅ Honest about being synthetic

## 📊 Comparison with Other Systems

| Feature | JARVIS-2v | Traditional LLM | RAG System |
|---------|-----------|-----------------|------------|
| Intelligence Location | Adapters | Weights | Embeddings |
| Learning Method | Add adapters | Fine-tune | Add documents |
| Explainability | High | Low | Medium |
| Catastrophic Forgetting | No | Yes | No |
| Inference Speed | Fast | Fast | Medium |
| Memory Usage | Low | High | Medium |
| Offline Capable | Yes | Yes | Partial |

## 🔮 Future Enhancements

Planned features:

1. **Adapter Pruning** - Remove low-performing adapters
2. **Automatic Bit Learning** - Learn bit patterns from data
3. **Multi-Model Support** - Different models for different domains
4. **Distributed Adapters** - Adapters across multiple nodes
5. **Active Learning** - Request specific training data
6. **Adapter Merging** - Combine related adapters

## 📚 Related Documentation

- [Training Guide](./TRAINING.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [API Reference](./API.md)
- [Ollama Integration](../ollama/README.md)

## 🎓 Research Papers

Key concepts:
- Parameter-Efficient Fine-Tuning (PEFT)
- Continual Learning
- Modular Neural Networks
- Explainable AI
- Edge AI Optimization

## 💡 Key Takeaways

1. **Intelligence ≠ Base Model** - JARVIS-2v's intelligence is in adapters, not weights
2. **Training = Creating Adapters** - Not fine-tuning, but building knowledge modules
3. **Non-Destructive** - All learning preserves existing knowledge
4. **Explainable** - Every decision has a clear path
5. **Modular** - Components can be swapped and upgraded independently

---

**JARVIS-2v**: Modular intelligence for the edge AI era.
