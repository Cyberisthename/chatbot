# JARVIS-2v – Modular Edge AI & Synthetic Quantum Lab

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)

**A low-power, modular AI engine with adapter-based memory, synthetic quantum-style experiment modules, and a visual lab UI – designed to run on local hardware and edge devices like Jetson Orin / FeatherEdge.**

---

## 🎯 NEW! One-Command Startup!

**Start JARVIS AI with ONE command:**

```bash
./run_ai.sh              # Linux/Mac
run_ai.bat              # Windows
```

That's it! The script will:
- ✅ Detect available AI backends (Ollama, Pinokio, Local)
- 📋 Show you a menu to choose from
- 🤖 Set up everything automatically
- 🌐 Launch beautiful web UI at http://localhost:3001

**📖 Read:** [START_HERE.md](START_HERE.md) | [GETTING_STARTED.md](GETTING_STARTED.md) | [BRANCH_README.md](BRANCH_README.md)

---

## 🚀 Quick Start

### 🎯 The Easy Way (Recommended for Everyone!)

**One script to run everything:**

```bash
# Linux/Mac
./run_ai.sh

# Windows
run_ai.bat
```

That's it! The script will:
- ✅ Detect available AI backends (Ollama, Pinokio, Local)
- 📋 Show you a menu to choose from
- 🤖 Set up everything automatically
- 🌐 Launch the web UI at http://localhost:3001

**Learn more:** See [README_EASY_RUN.md](README_EASY_RUN.md) and [QUICKSTART_LOCAL_AI.md](QUICKSTART_LOCAL_AI.md)

### 🔧 Manual Setup (Advanced)

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Run in standard mode
./scripts/start_local.sh

# Run on Jetson Orin
./scripts/start_jetson.sh --low-power

# Run in offline mode (no network)
./scripts/start_jetson.sh --offline
```

## 🧠 Core Architecture

### Modular Adapter System
- **Non-destructive learning**: New adapters created for each task, old ones frozen
- **Y/Z/X bit routing**: 16/8/8 bit vectors for task classification and routing
- **Graph-based relationships**: Adapters with parent/child dependencies
- **Explainable routing**: Every decision logged with bit patterns and reasoning

### Synthetic Quantum Module
- **Real artifacts**: Interference experiments, Bell pair simulations, CHSH tests
- **Adapter linkage**: Each artifact creates linked adapters for learned patterns
- **Reusable context**: Artifacts can be replayed and used as context for queries
- **Honest simulation**: All artifacts labeled as synthetic with full metadata

### Edge-Ready Design
- **Jetson optimization**: CUDA layers, memory management, power profiles
- **Offline operation**: Full functionality without network access
- **Resource profiles**: `low_power`, `standard`, `jetson_orin` modes
- **Stable API contract**: Simple endpoints for satellite/robot integration

## 📁 Project Structure

```
javis-2v/
├── config.yaml              # Main configuration
├── config_jetson.yaml       # Jetson-specific config
├── src/
│   ├── __init__.py
│   ├── core/                # Core adapter engine
│   │   └── adapter_engine.py
│   ├── api/                 # FastAPI server
│   │   └── main.py
│   ├── quantum/            # Synthetic quantum module
│   │   └── synthetic_quantum.py
│   └── ui/                  # React dashboard
│       └── package.json
├── scripts/                 # Deployment scripts
│   ├── start_local.sh
│   └── start_jetson.sh
├── adapters/                # Adapter storage
├── quantum_artifacts/       # Quantum artifacts
├── models/                  # GGUF models
└── tests/                   # Test suite
```

## 🔧 Configuration

### Core Settings (`config.yaml`)

```yaml
engine:
  name: "JARVIS-2v"
  mode: "standard"  # low_power | standard | jetson_orin

model:
  path: "./models/jarvis-7b-q4_0.gguf"
  gpu_layers: 0     # 0 for CPU, 30 for Jetson
  device: "cpu"     # cpu | cuda | jetson

adapters:
  auto_create: true
  freeze_after_creation: true

edge:
  low_power_mode: false
  offline_mode: false
```

## 🎛️ API Endpoints

### Main Chat
```http
POST /chat
{
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "options": {"temperature": 0.7}
}
```

### Adapter Management
```http
POST /adapters
{
  "task_tags": ["math", "physics"],
  "parameters": {"complexity": "high"}
}

GET /adapters
```

### Quantum Experiments
```http
POST /quantum/experiment
{
  "experiment_type": "interference_experiment",
  "config": {"iterations": 1000, "noise_level": 0.1}
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run adapter routing tests
python -m tests.test_adapter_routing

# Run quantum simulation tests
python -m tests.test_quantum_artifacts
```

## 📊 Benchmarking

### Performance Metrics
- Time per request: ~100ms (CPU) / ~50ms (Jetson GPU)
- Adapters per request: 1-3 average
- Memory usage: 500MB (low power) / 2GB (standard)
- Power profile: Low (5W) / Standard (15W) / Jetson (25W)

### Continual Learning Tests
```bash
# Test non-destructive learning
python scripts/test_continual_learning.py

# Verify adapter isolation
python scripts/verify_adapter_isolation.py
```

## 🛠️ Deployment

### 🌐 Web Deployment (Vercel/Netlify)

**For a clean, lightweight web UI deployment without model files:**

```bash
# Use the deployment branch
git checkout deploy/vercel-clean-webapp-no-lfs

# Deploy to Vercel (recommended)
# 1. Connect your GitHub repo to Vercel
# 2. Select branch: deploy/vercel-clean-webapp-no-lfs
# 3. Click Deploy - that's it!
```

📖 **Detailed Guide**: See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for complete instructions

**What you get:**
- ✅ Beautiful web chat interface
- ✅ No Git LFS issues
- ✅ Fast deployment (~30-60 seconds)
- ✅ Works on Vercel's free tier
- ✅ Demo/mock AI responses (no model files needed)

**Branch Overview:**
- `main` - Full system with models (for local development)
- `deploy/vercel-clean-webapp-no-lfs` - Web UI only (for cloud deployment)

### Jetson Orin NX Deployment
```bash
# 1. Install Jetson-specific dependencies
./scripts/start_jetson.sh --install

# 2. Optimize GPU layers
export JARVIS_GPU_LAYERS=30

# 3. Run with Jetson config
./scripts/start_jetson.sh --config config_jetson.yaml
```

### Docker Deployment
```bash
docker build -f Dockerfile.jetson -t jarvis-2v:jetson .
docker run --gpus all -p 3001:3001 jarvis-2v:jetson
```

## 🔬 Research Applications

### Satellite Systems (FeatherEdge/FlatSat)
- Offline operation for in-orbit deployment
- Low-power profile for power constraints
- Compact memory footprint (6GB RAM limit)
- Stable API for flight OS integration

### Robotics Integration
- Real-time adapter routing for task switching
- Quantum artifact usage for complex decision making
- Non-destructive learning for continuous adaptation

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [Adapter System Deep Dive](docs/adapters.md)
- [Quantum Module Guide](docs/quantum.md)
- [Edge Deployment Guide](docs/edge-deploy.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- llama.cpp team for GGUF model support
- NVIDIA Jetson team for edge AI tools
- Synthetic quantum research community
- Ben (J.A.R.V.I.S. creator) for the original vision

---

**JARVIS-2v: Because even AIs deserve a modular, quantum-enhanced future.**