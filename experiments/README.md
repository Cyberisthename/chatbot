# Quantum Experiments Collection

This directory contains standalone quantum physics experiments that demonstrate key concepts in quantum mechanics.

## Available Experiments

### 1. Quick Interference Experiment (`quick_interference.py`)

**What it does:**
- Simulates a classic double-slit interference pattern
- Demonstrates wave-particle duality
- Shows bright and dark fringes caused by quantum interference

**Physics Concept:**
This is one of the most famous experiments in quantum mechanics. It shows that particles (like photons or electrons) can behave like waves and interfere with themselves when passing through two slits. The resulting pattern of bright and dark fringes proves the wave nature of quantum objects.

**How to run:**
```bash
python3 experiments/quick_interference.py
```

**Output:**
- `artifacts/interference_pattern.png` - Visual plot of the interference pattern
- `artifacts/interference_result.json` - Quantitative measurements

**Example Output:**
```json
{
  "experiment": "quantum_double_slit_interference",
  "params": {
    "n_points": 2048,
    "wavelength": 1.0,
    "slit_distance": 5.0,
    "slit_width": 0.5,
    "L": 50.0
  },
  "results": {
    "mean_intensity": 0.024,
    "max_intensity": 1.0,
    "min_intensity": 7.26e-33,
    "std_intensity": 0.126
  }
}
```

**Customization:**
You can modify the parameters in the `run_interference()` function:
- `n_points`: Resolution of the detector (default: 2048)
- `wavelength`: Wavelength of the particle (default: 1.0)
- `slit_distance`: Distance between the two slits (default: 5.0)
- `slit_width`: Width of each slit (default: 0.5)
- `L`: Distance from slits to detector screen (default: 50.0)

**Educational Value:**
Perfect for showing to:
- Students learning quantum mechanics
- Anyone curious about wave-particle duality
- Demonstrations of quantum interference
- Understanding why quantum mechanics is "weird"

## Related Experiments

The following quantum experiments are also available in the root directory:

### Quantum Uncertainty Collapse
- **Files:** `uncertainty_experiment.py`, `uncertainty_experiment_headless.py`
- **Concept:** Quantum decoherence and collapse of uncertainty
- **Documentation:** See `UNCERTAINTY_EXPERIMENT_README.md`

## Adding New Experiments

When creating new experiments, please:

1. **Follow the naming convention:** `descriptive_name.py`
2. **Include a docstring** explaining the physics concept
3. **Save artifacts** to the `artifacts/` directory
4. **Use JSON format** for data with structure:
   ```python
   {
       "experiment": "experiment_name",
       "version": "1.0",
       "params": {...},
       "results": {...},
       "physics": {...},
       "notes": "..."
   }
   ```
5. **Generate visualizations** when applicable (PNG format)
6. **Update this README** with the new experiment

## Requirements

All experiments require:
- Python 3.8+
- numpy >= 1.24.0
- matplotlib >= 3.7.0

Install with:
```bash
pip install -r requirements.txt
```

Or in the virtual environment:
```bash
source .venv/bin/activate
pip install numpy matplotlib
```

## Project Integration

These experiments integrate with the broader J.A.R.V.I.S. AI System project and complement the Quantacap quantum computing simulations. All experiments save artifacts to the shared `artifacts/` directory for easy analysis and comparison.

---

**Last Updated:** October 2024
