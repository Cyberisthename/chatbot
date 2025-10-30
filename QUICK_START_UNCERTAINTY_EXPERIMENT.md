# Quick Start: Quantum Uncertainty Collapse Experiment

## 🚀 Run the Experiment

### Option 1: Headless (Recommended for First Run)
```bash
python3 uncertainty_experiment_headless.py
```
**Output:**
- `artifacts/uncertainty_experiment.json` - Data results
- `artifacts/uncertainty_experiment.png` - Visualization

### Option 2: Interactive Animation
```bash
python3 uncertainty_experiment.py
```
**Note:** Requires display. Close the animation window to save artifacts.

## 📊 View Results

### Check the JSON data:
```bash
cat artifacts/uncertainty_experiment.json | python3 -m json.tool
```

### View the visualization:
```bash
# Linux
xdg-open artifacts/uncertainty_experiment.png

# macOS
open artifacts/uncertainty_experiment.png

# Windows
start artifacts/uncertainty_experiment.png
```

## 🔧 Customize Parameters

Edit the parameters at the top of either script:

```python
n_points = 256      # Spatial resolution (more = smoother)
frames = 200        # Animation length (more = longer convergence)
decay = 0.98        # Decoherence rate (0.9-0.99, higher = slower)
amplitude = 1.0     # Initial noise level
```

## 📖 Learn More

See [UNCERTAINTY_EXPERIMENT_README.md](UNCERTAINTY_EXPERIMENT_README.md) for:
- Physics explanation
- Detailed parameter guide
- Output interpretation
- Mathematical formulation

## 🧪 What's Happening?

This experiment simulates **quantum decoherence**:

1. Start with a quantum wave packet: ψ(x) = e^(-x²)
2. Add random noise (quantum uncertainty)
3. Watch it gradually stabilize (collapse to classical state)
4. Measure the final probability distribution

**Result:** You see chaos becoming order - a key concept in quantum mechanics!

## 💡 Quick Tips

- **Fast test**: Set `frames = 50, decay = 0.95` for quick results
- **High quality**: Set `n_points = 512, frames = 500` for smooth animation
- **Different behavior**: Try `decay = 0.90` for rapid collapse or `0.99` for slow evolution

---

**Duration:** ~5 seconds (headless) or ~10 seconds (interactive)
**Requirements:** Python 3.8+, numpy, matplotlib
