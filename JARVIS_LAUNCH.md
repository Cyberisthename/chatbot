# 🚀 J.A.R.V.I.S. Personal Assistant - Launch Guide

## Quick Start

### Option 1: Simple Launch (Recommended)
```bash
# Using the virtual environment
source .venv/bin/activate
python jarvis_assistant.py
```

Then open your browser to: **http://localhost:7860**

### Option 2: Using the Launcher Script
```bash
source .venv/bin/activate
python launch_jarvis.py
```

## Features Available

### 💬 Neural Chat
- Quantum LLM-powered conversation
- Real-time quantum telemetry (coherence, entanglement, interference, fidelity)
- Adjustable temperature and token depth
- Conversation history

### 🔣 Thought-Compression Language (TCL)
- Process TCL expressions for enhanced cognition
- Compress concepts into symbolic form
- Causal chain analysis
- Enhanced reasoning

### 🔬 Cancer Research
- Generate cancer treatment hypotheses
- Analyze protein quantum properties
- Access biological knowledge base
- Quantum-sensitive pathway analysis

### 🌌 Multiversal Computing
- Parallel universe protein folding
- Real physics-based computation
- Cross-universe interference analysis

### ⚛️ Quantum Lab
- Run quantum experiments
- Bell pair simulations
- CHSH tests
- Negative information experiments

### 🔌 Adapter Engine
- Create and manage adapters
- YZX bit pattern routing
- Task specialization

### 📊 System Status
- View all system statuses
- Model architecture info
- Component health monitoring

## System Requirements

- Python 3.10+
- NumPy
- Gradio
- FastAPI (for API routes)
- PyYAML

## Installation

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install numpy gradio fastapi uvicorn pyyaml

# Launch JARVIS
python jarvis_assistant.py
```

## Architecture

The JARVIS Assistant integrates:

1. **Quantum LLM** - Transformer with quantum-inspired attention (12M+ parameters)
2. **TCL Engine** - Thought-Compression Language for cognitive enhancement
3. **Cancer Research** - Biological knowledge + quantum H-bond analysis
4. **Multiversal Computing** - Parallel universe protein folding
5. **Quantum Engine** - Synthetic quantum experiments
6. **Adapter Engine** - Modular task routing

## Troubleshooting

### Import Errors
Make sure you're using the virtual environment:
```bash
source .venv/bin/activate
```

### Port Already in Use
Change the port in `jarvis_assistant.py`:
```python
demo.launch(server_port=7861)  # Use different port
```

### Model Not Loading
The system will initialize a fresh model if the trained model file is not found. To use the trained model, ensure these files exist:
- `ready-to-deploy-hf/jarvis_quantum_llm.npz`
- `ready-to-deploy-hf/tokenizer.json`

## License

See LICENSE file for details.

---

**J.A.R.V.I.S. - Just A Rather Very Intelligent System**
