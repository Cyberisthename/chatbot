# Quick Start Guide - Quantum Experiments

## 🚀 Run Experiments

### Double-Slit Interference
```bash
python3 experiments/quick_interference.py
```
**Shows:** Wave-particle duality with bright/dark fringes

### Quantum Uncertainty Collapse
```bash
python3 uncertainty_experiment_headless.py
```
**Shows:** Decoherence and quantum-to-classical transition

## 📊 View Results

All experiments save to `artifacts/`:

```bash
# List all results
ls -lh artifacts/

# View specific results
cat artifacts/interference_result.json
cat artifacts/uncertainty_experiment.json

# View images (Linux)
xdg-open artifacts/interference_pattern.png
xdg-open artifacts/uncertainty_experiment.png
```

## 🔧 Requirements

```bash
# With virtual environment (recommended)
source .venv/bin/activate
python3 experiments/quick_interference.py

# Or install dependencies
pip install numpy matplotlib
```

## 📖 Learn More

- **Interference Details:** [INTERFERENCE_EXPERIMENT.md](INTERFERENCE_EXPERIMENT.md)
- **Uncertainty Details:** [../UNCERTAINTY_EXPERIMENT_README.md](../UNCERTAINTY_EXPERIMENT_README.md)
- **All Experiments:** [README.md](README.md)

---

**Tip:** Start with `quick_interference.py` - it runs in ~1 second!
