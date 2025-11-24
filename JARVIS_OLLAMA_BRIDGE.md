# Jarvis + Ollama API Bridge

This integration wires **Jarvis-2v** (quantum phase detector lab) with **Ollama** (local LLM runtime) so that language models can execute quantum experiments via natural language.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Ollama LLM     │  ←───→  │  chat_with_lab   │  ←───→  │  jarvis_api.py   │
│  (Language)     │         │  (Bridge)        │         │  (Lab Engine)    │
└─────────────────┘         └──────────────────┘         └──────────────────┘
     llama3.1                  Custom tool                  FastAPI + Jarvis
     or ben-lab                calling loop                 PhaseDetector
```

- **Jarvis** = Lab engine (quantum phase simulator)
- **Ollama** = Local LLM brain (language understanding)
- **Bridge** = Tool-calling glue code

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama** installed and running:
   ```bash
   # Install from https://ollama.ai
   ollama pull llama3.1  # or your custom model
   ```

### Step 1: Start the Jarvis Lab API

In one terminal:

```bash
python jarvis_api.py
```

Output:
```
🚀 Starting Jarvis Lab API...
📡 Endpoints available at http://127.0.0.1:8000
📚 API docs at http://127.0.0.1:8000/docs
🔬 Quantum detector initialized with 1 devices
```

The lab is now live at **http://127.0.0.1:8000**.

### Step 2: Chat with the Lab

In another terminal:

```bash
python chat_with_lab.py
```

Output:
```
🤖 Connected to Jarvis Lab API at http://127.0.0.1:8000
🧠 Using Ollama model: llama3.1
Type 'exit' or 'quit' to stop.

Ben:
```

### Step 3: Try It Out

Example conversation:

```
Ben: Run an Ising experiment with bias 0.7 and explain what TRI means.

AI: TOOL: {"name": "run_phase", "args": {"phase_type": "ising_symmetry_breaking", "system_size": 32, "depth": 8, "seed": 42, "bias": 0.7}}

AI (after lab): The experiment completed successfully. Ising symmetry breaking phase detected with experiment ID abc123...
TRI (Time-Reversal Instability) measures how different a quantum system behaves when you reverse time...
```

---

## 🔬 Lab API Endpoints

The Jarvis Lab API exposes these tools:

### 1. `/run_phase_experiment`

Run a single quantum phase experiment.

**Parameters:**
- `phase_type`: `ising_symmetry_breaking`, `spt_cluster`, `trivial_product`, or `pseudorandom`
- `system_size`: System size (4-256, default 32)
- `depth`: Circuit depth (1-32, default 8)
- `seed`: Random seed for reproducibility
- `bias`: Optional bias parameter (0.0-1.0)

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/run_phase_experiment \
  -H "Content-Type: application/json" \
  -d '{"phase_type": "ising_symmetry_breaking", "system_size": 32, "depth": 8, "seed": 42, "bias": 0.7}'
```

**Response:**
```json
{
  "experiment_id": "abc123...",
  "phase_type": "ising_symmetry_breaking",
  "feature_vector": [0.521, 0.387, ...],
  "summary": {"entropy": 0.521, ...},
  "params": {...}
}
```

### 2. `/tri`

Time-Reversal Instability test (forward + reverse bias).

**Parameters:**
- `phase_type`: Phase type (default: `ising_symmetry_breaking`)
- `system_size`: System size (default: 32)
- `depth`: Circuit depth (default: 8)
- `bias`: Forward bias (default: 0.7)
- `seed`: Random seed (default: 42)

**Response:**
```json
{
  "forward_id": "...",
  "reverse_id": "...",
  "TRI": 1.234,
  "params": {...}
}
```

### 3. `/discovery`

Unsupervised phase discovery using k-means clustering.

**Parameters:**
- `phases`: List of phase types to cluster
- `num_per_phase`: Samples per phase (5-100, default 20)
- `k`: Number of clusters (defaults to `len(phases)`)
- `iterations`: K-means iterations (10-100, default 25)

**Response:**
```json
{
  "cluster_label_stats": [
    {"ising_symmetry_breaking": 28, "spt_cluster": 2},
    ...
  ],
  "num_samples": 80,
  "num_clusters": 4
}
```

### 4. `/replay_drift`

Replay drift scaling experiment (depth scaling).

**Parameters:**
- `phase_type`: Phase type
- `system_size`: System size
- `base_depth`: Base circuit depth
- `seed`: Random seed
- `depth_factors`: List of depth multipliers (e.g., `[1, 2, 3]`)

**Response:**
```json
{
  "phase_type": "ising_symmetry_breaking",
  "base_depth": 6,
  "runs": [
    {"depth": 6, "drift": 0.0, "id": "...", "features": [...]},
    {"depth": 12, "drift": 1.234, ...},
    ...
  ]
}
```

---

## 🤖 Tool Calling Format

The bridge script (`chat_with_lab.py`) uses a simple custom tool-calling protocol.

When the LLM wants to execute a lab experiment, it responds with:

```
TOOL: {"name": "tool_name", "args": {...}}
```

Available tools:
- `run_phase` → `/run_phase_experiment`
- `tri` → `/tri`
- `discovery` → `/discovery`
- `replay_drift` → `/replay_drift`

**Example:**

```python
TOOL: {"name": "tri", "args": {"phase_type": "spt_cluster", "bias": 0.6, "seed": 123}}
```

The bridge:
1. Parses the `TOOL:` line
2. Calls the corresponding Jarvis API endpoint
3. Feeds the result back to the LLM
4. LLM interprets the result and responds to the user

---

## 🧪 Test Suite

Run the discovery suite to verify Jarvis is working:

```bash
python experiments/discovery_suite.py
```

Expected output:
```
=== EXPERIMENT A: Time-Reversal Instability ===
ising_symmetry_breaking   TRI = 0.523456
spt_cluster              TRI = 0.123456
...

=== EXPERIMENT B: Unsupervised Phase Discovery ===
Cluster 0: {'ising_symmetry_breaking': 28, 'spt_cluster': 2}
...
```

---

## 📦 Fine-Tuning Your Own Model

Once you collect enough Q&A pairs from your experiments, you can fine-tune a custom model (see `OLLAMA_FINETUNE_GUIDE.md`).

**Example workflow:**

1. Collect logs from experiments:
   ```bash
   python experiments/build_phase_dataset.py --num-per-phase 50
   ```

2. Generate training data:
   ```bash
   python generate_lab_training_data.py
   ```

3. Fine-tune with Ollama:
   ```bash
   cd ollama/
   ollama create ben-lab -f Modelfile
   ```

4. Use your custom model:
   ```python
   # In chat_with_lab.py, change DEFAULT_MODEL
   DEFAULT_MODEL = "ben-lab"
   ```

---

## 🔧 Configuration

### Change Ollama Model

Edit `chat_with_lab.py`:

```python
DEFAULT_MODEL = "llama3.1"  # or "ben-lab", "mistral", etc.
```

### Change Lab Server Port

Edit `jarvis_api.py`:

```python
uvicorn.run(app, host="127.0.0.1", port=8000)  # change port
```

And update `chat_with_lab.py`:

```python
JARVIS_LAB_URL = "http://127.0.0.1:8000"
```

### Add More Devices

Edit `jarvis_api.py`:

```python
devices = [
    AdapterDevice(
        id="quantum_0",
        label="Quantum Simulator",
        kind=DeviceKind.VIRTUAL,
        perf_score=50.0,
        max_concurrency=8,
        capabilities={OperationKind.QUANTUM},
    ),
    # Add more devices here
]
```

---

## 🐛 Troubleshooting

### Jarvis API won't start

**Error:** `ModuleNotFoundError: No module named 'jarvis5090x'`

**Fix:**
```bash
pip install -r requirements.txt
```

### Ollama connection refused

**Error:** `requests.exceptions.ConnectionError: ... 11434 ...`

**Fix:**
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify:
ollama list
```

### Lab API not reachable

**Error:** `Jarvis Lab API is not reachable`

**Fix:**
```bash
# Start the lab API in one terminal
python jarvis_api.py

# Wait for "Uvicorn running on http://127.0.0.1:8000"
# Then run chat_with_lab.py in another terminal
```

### Model not found

**Error:** `Ollama returned status 404: model 'ben-lab' not found`

**Fix:**
```bash
# Pull a model first
ollama pull llama3.1

# Or create your custom model
cd ollama/
ollama create ben-lab -f Modelfile
```

---

## 📚 Related Documentation

- **Jarvis-2v**: [PHASE_DETECTOR.md](PHASE_DETECTOR.md), [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)
- **Ollama Fine-tuning**: [OLLAMA_FINETUNE_GUIDE.md](OLLAMA_FINETUNE_GUIDE.md)
- **RL Scientist**: [README.md](README.md) (Phase MLP + RL section)
- **Discovery Suite**: [experiments/discovery_suite.py](experiments/discovery_suite.py)

---

## 🎯 What's Next?

1. **Run experiments** via chat interface
2. **Collect logs** of successful experiments
3. **Generate training data** from logs + feature vectors
4. **Fine-tune** a small LLM (e.g., Qwen2.5-1.5B or Llama-3.2-1B)
5. **Deploy** as `ben-lab` in Ollama
6. **Iterate**: Your LLM gets smarter about phase physics!

This is the foundation of a self-improving lab AI. 🚀

---

## 📝 License

Same as the Jarvis-2v project (see [LICENSE](LICENSE)).
