# Quantum-Style Uncertainty Collapse Experiment

## Overview

This experiment demonstrates how random noise (chaos) gradually collapses into a stable quantum-like amplitude distribution — a visual "quantum stabilization" demo.

## What It Does

1. **Starts with a Gaussian wave packet**: ψ(x) = e^{-x²}
2. **Adds random noise each frame** to simulate "quantum uncertainty"
3. **Gradually damps the noise** with `amplitude *= decay`
4. **Shows a smooth collapse** into a stable probability distribution

## Physics Concept

This is a digital analog of **decoherence** — how chaos (quantum randomness) stabilizes into classical order. The animation shows how amplitude² converges to a normal distribution while losing phase variation.

## How to Run

There are two versions of this experiment:

### Interactive Version (with animation)

```bash
python3 uncertainty_experiment.py
```

This will:
- Display an animated visualization showing the collapse process in real-time
- Show the current randomness level in the title as it updates
- Automatically save artifacts to `artifacts/uncertainty_experiment.json` when you close the window

### Headless Version (for CI/CD and batch processing)

```bash
python3 uncertainty_experiment_headless.py
```

This will:
- Run the experiment without requiring a display
- Generate a static visualization showing the evolution over time
- Save both `artifacts/uncertainty_experiment.json` and `artifacts/uncertainty_experiment.png`
- Perfect for automated testing, CI/CD pipelines, or remote servers

### What You'll See

- A cyan line showing the probability distribution |ψ|²
- The title updates to show:
  - Current frame number
  - Current randomness level (amplitude)
- The distribution starts noisy and stabilizes over 200 frames

## Output Artifacts

When the experiment completes, the scripts generate:

### JSON Data File

**`artifacts/uncertainty_experiment.json`** containing:
- `final_entropy`: Variance of the final probability distribution
- `final_amplitude`: How much randomness remains
- `position_mean`: Expected position value
- `position_std`: Position uncertainty (standard deviation)
- Various other statistical measures

### Visualization Image (headless version only)

**`artifacts/uncertainty_experiment.png`** showing:
- Top panel: Evolution of probability distribution across multiple frames
- Bottom panel: Exponential decay of noise amplitude over time
- Color-coded by frame progression (early frames in dark blue, later frames in bright yellow)

### Example JSON Output

```json
{
  "experiment": "quantum_uncertainty_collapse",
  "version": "1.0",
  "parameters": {
    "n_points": 256,
    "frames": 200,
    "decay_rate": 0.98,
    "initial_amplitude": 1.0
  },
  "results": {
    "final_amplitude": 0.017588,
    "final_entropy": 0.085978,
    "final_mean": 0.156049,
    "final_std": 0.293221,
    "position_mean": -0.000014,
    "position_std": 0.500179,
    "max_amplitude": 1.002695,
    "min_amplitude": 3.993673e-11
  },
  "physics": {
    "interpretation": "Digital analog of quantum decoherence",
    "process": "Chaos (quantum randomness) stabilizing into classical order",
    "convergence": "Amplitude² converges to Gaussian distribution",
    "decoherence_rate": "2.0% per frame"
  },
  "notes": "Simulates decoherence: chaos stabilizing into classical order"
}
```

## Customization

Edit the parameters at the top of `uncertainty_experiment.py`:

```python
n_points = 256      # Number of spatial points
frames = 200        # Number of animation frames
decay = 0.98        # How fast randomness stabilizes (0.9-0.99)
amplitude = 1.0     # Initial noise amplitude
```

### Parameter Effects

- **Higher `decay`** (closer to 1.0): Slower stabilization, more frames to watch
- **Lower `decay`** (0.9 or less): Faster collapse, reaches stability quickly
- **More `frames`**: Longer animation, shows more of the convergence process
- **More `n_points`**: Smoother curves, but slower rendering

## Talk Points

If you show this to someone:

> "This is a digital analog of decoherence — how chaos (quantum randomness) 
> stabilizes into classical order. The animation shows how amplitude² converges 
> to a normal distribution while losing phase variation."

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- matplotlib >= 3.7.0

Install with:

```bash
pip install numpy matplotlib
```

Or use the project requirements:

```bash
pip install -r requirements.txt
```

## Technical Details

### Wave Function Evolution

At each frame `t`, the wave function evolves as:

```
ψ(x, t) = e^{-x²} · e^{iθ(t)} + A(t) · η(t)
```

Where:
- `e^{-x²}`: Base Gaussian wave packet
- `e^{iθ(t)}`: Phase evolution (θ = t/10)
- `A(t)`: Amplitude that decays as A(t+1) = 0.98 × A(t)
- `η(t)`: Random noise from standard normal distribution

### Probability Distribution

The measured quantity is:

```
P(x, t) = |ψ(x, t)|²
```

Normalized for display as:

```
P_normalized(x, t) = P(x, t) / max(P(x, t))
```

## Integration with Other Experiments

This experiment follows the same artifact structure as other quantum experiments in the project:

- Saves to `artifacts/` directory
- Uses JSON format for results
- Can be summarized with `summarize_quantum_artifacts.py`

---

**Created as part of the J.A.R.V.I.S. AI System quantum experiments collection**
