# Quick Quantum Interference Experiment

## Overview

This experiment simulates the famous **double-slit interference pattern** that demonstrates one of the most fundamental and mysterious aspects of quantum mechanics: **wave-particle duality**.

## What Makes This Special?

The double-slit experiment is often called "the only mystery" of quantum mechanics because it shows that:

1. **Particles behave like waves** - They can be in multiple places at once
2. **Interference happens** - Waves from two slits combine to create bright and dark fringes
3. **Observation matters** - When you measure which slit a particle goes through, the pattern disappears

## The Physics

### Mathematical Model

For two slits separated by distance `d`, the amplitude at position `x` on the detector screen is:

```
ψ(x) = ψ₁(x) + ψ₂(x)
```

Where:
- `ψ₁(x)` = amplitude from slit 1 (sinc function)
- `ψ₂(x)` = amplitude from slit 2 (sinc function with phase shift)

The intensity pattern is:
```
I(x) = |ψ(x)|²
```

This creates the characteristic bright and dark fringes.

### Key Parameters

- **Wavelength (λ)**: Smaller wavelength = narrower fringes
- **Slit Distance (d)**: Larger distance = more fringes in the pattern
- **Slit Width (w)**: Affects the envelope of the pattern
- **Screen Distance (L)**: Distance from slits to detector

### Fringe Spacing

The spacing between bright fringes is approximately:
```
Δx ≈ λL / d
```

## Running the Experiment

### Basic Usage

```bash
cd /home/engine/project
python3 experiments/quick_interference.py
```

### Output Files

**JSON Data (`artifacts/interference_result.json`):**
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
    "min_intensity": ~0.0,
    "std_intensity": 0.126
  },
  "physics": {
    "phenomenon": "wave-particle duality",
    "description": "Bright and dark fringes from quantum interference"
  }
}
```

**Visualization (`artifacts/interference_pattern.png`):**
- Blue curve showing normalized intensity
- Clear peaks (bright fringes) and valleys (dark fringes)
- Grid for easy reading of values

## Customizing the Experiment

Edit `quick_interference.py` to modify the `run_interference()` call:

```python
# Example: Wider slit separation (more fringes)
result = run_interference(
    n_points=2048,
    wavelength=1.0,
    slit_distance=10.0,  # Changed from 5.0
    slit_width=0.5,
    L=50.0
)

# Example: Shorter wavelength (tighter fringes)
result = run_interference(
    n_points=2048,
    wavelength=0.5,  # Changed from 1.0
    slit_distance=5.0,
    slit_width=0.5,
    L=50.0
)
```

## What You See in the Plot

### Bright Fringes (Peaks)
- Occur where waves from both slits arrive **in phase**
- Path difference = integer multiple of wavelength
- **Constructive interference**: amplitudes add

### Dark Fringes (Valleys)
- Occur where waves arrive **out of phase** 
- Path difference = half-integer multiple of wavelength
- **Destructive interference**: amplitudes cancel

### Envelope Function
- The overall shape modulated by the sinc function
- Represents diffraction from individual slits
- Makes outer fringes gradually dimmer

## Historical Context

### Original Experiment (Thomas Young, 1801)
- Used sunlight and two narrow slits
- Proved light was a wave (controversial at the time)
- Contradicted Newton's particle theory of light

### Quantum Version (20th Century)
- Done with individual photons, electrons, even molecules
- Shows that **single particles** create interference patterns
- Particles somehow "interfere with themselves"
- Led to development of quantum mechanics

### Modern Applications
- Quantum computing: superposition and interference
- Quantum cryptography: using quantum properties for security
- Quantum sensors: ultra-precise measurements
- Fundamental tests of quantum theory

## Talk Points

When showing this to someone:

> "This pattern proves that quantum particles behave like waves. Even if you 
> send particles one at a time, they still create this pattern - meaning each 
> particle somehow goes through both slits and interferes with itself. This is 
> the mystery at the heart of quantum mechanics."

## Try These Variations

### More Fringes
```python
run_interference(slit_distance=15.0)  # Triple the separation
```

### Finer Detail
```python
run_interference(n_points=4096)  # Double the resolution
```

### Different Wavelength Regime
```python
run_interference(wavelength=2.0)  # Larger wavelength (infrared-like)
run_interference(wavelength=0.3)  # Smaller wavelength (UV-like)
```

### Narrower Slits
```python
run_interference(slit_width=0.2)  # Narrower slits = wider diffraction
```

## Integration with Other Experiments

This experiment complements:
- **Uncertainty Collapse** - Shows wave-like behavior before measurement
- **CHSH Experiments** - Related quantum correlations
- **Atom Simulations** - Wave functions in quantum systems

All use the same artifact structure for easy comparison and analysis.

## Further Reading

### Papers & Books
- Feynman Lectures Vol. III, Chapter 1 - "Quantum Behavior"
- "The Feynman Lectures on Physics" - The classic explanation
- Original Young's paper (1801) - Historical perspective

### Online Resources
- Physics Stack Exchange: quantum interference
- YouTube: "The Quantum Double Slit Experiment"
- MIT OpenCourseWare: 8.04 Quantum Physics I

---

**Experiment Duration:** ~1 second  
**Output Size:** ~55 KB (PNG) + ~631 bytes (JSON)  
**Requirements:** Python 3.8+, numpy, matplotlib
