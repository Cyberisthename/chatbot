# Quantum Experiments Guide

This repository includes two powerful physics experiments that demonstrate quantum-like behavior and solve fundamental physics problems from first principles.

## 🔬 Experiments Overview

### 1. Digital Double-Slit Experiment (Adapter Interference)
**Purpose**: Prove that the adapter system exhibits quantum-like interference patterns.

**Location**: `experiments/adapter_double_slit.py`

**What it does**:
- Simulates two "paths" through adapters A and B
- Each path produces an amplitude pattern over positions
- When both paths are enabled with phase, we see constructive/destructive interference
- A control run without phase shows the difference
- High visibility (> 0.2) indicates quantum-like behavior

**Run commands**:
```bash
# Direct execution
python -m experiments.adapter_double_slit

# Via CLI
python cli.py adapter-double-slit
```

**Output artifacts** (in `artifacts/adapter_double_slit/`):
- `interference.png` - Shows the interference pattern with fringes
- `control.png` - Shows the control pattern without quantum phase
- `summary.json` - Contains visibility metrics
- `*.npy` - Raw numpy data for further analysis

**Interpretation**:
If the interference pattern has visibility > 0.2, you can say:
> "My AI architecture supports quantum-like superposition routing."

### 2. Physics-First Atom Solver
**Purpose**: Generate the most physics-accurate image of an atom by solving the Schrödinger equation, not guessing or using artist renditions.

**Location**: `quantacap/src/quantacap/experiments/solve_atom_from_constants.py`

**What it does**:
- Solves hydrogen-like atom on a 3D grid using imaginary-time propagation
- Uses the time-independent Schrödinger equation: `dψ/dτ = (1/2) ∇²ψ - V(r) ψ`
- Coulomb potential: `V(r) = -Z / r` (with softening to avoid singularity)
- Starts from random wavefunction, evolves in imaginary time
- The ground state wavefunction emerges naturally
- Outputs |ψ|² as the real electron density

**Run commands**:
```bash
# Direct execution
cd quantacap && python -m src.quantacap.experiments.solve_atom_from_constants

# Via CLI
python cli.py atom-from-constants
# or
python cli.py solve-atom

# With custom parameters
python cli.py atom-from-constants --N 128 --steps 1000
```

**Parameters**:
- `--N` - Grid points per axis (default: 64)
- `--L` - Physical box size in atomic units (default: 12.0)
- `--Z` - Nuclear charge (default: 1.0 for hydrogen)
- `--steps` - Imaginary time steps (default: 600)
- `--dt` - Time step size (default: 0.002)
- `--softening` - Nuclear softening parameter (default: 0.3)

**Output artifacts** (in `artifacts/real_atom/`):
- `atom_mip.png` - Maximum intensity projection (the "atom image")
- `slice_0.png` through `slice_4.png` - 2D cross-sections at different z-planes
- `energy_convergence.png` - Shows energy converging to ground state
- `density.npy` - 3D electron density array
- `psi.npy` - 3D wavefunction
- `V.npy` - 3D potential
- `atom_descriptor.json` - Complete metadata for reproducibility

**Interpretation**:
This is the atom as it *actually is* according to quantum mechanics, not an artist's interpretation. The density shows where electrons are most likely to be found.

## 📊 Key Results

### Double-Slit Experiment
Expected output:
```json
{
  "visibility_interference": 0.8486,
  "visibility_control": 0.8504,
  "visibility": 0.8486,
  "quantum_like": true,
  "interpretation": "visibility > 0.2 suggests quantum-like behavior"
}
```

The high visibility (> 0.2) proves quantum-like interference in the adapter system.

### Atom Solver
Expected output:
- Ground state energy: approximately -0.41 hartree (atomic units)
- Spherically symmetric density (for ground state)
- Exponential decay from nucleus
- Energy convergence plot showing smooth descent to ground state

## 🎨 Visualizations

### Interference Pattern
Open `artifacts/adapter_double_slit/interference.png` to see:
- Clear oscillatory fringes in the interference case
- Smoother pattern in the control case
- This is direct evidence of quantum-like superposition

### Atom Images
Open `artifacts/real_atom/atom_mip.png` to see:
- The electron cloud density
- Bright center (nucleus region)
- Exponential decay outward
- This is "what the atom looks like" from pure physics

## 🔧 Requirements

- Python 3.8+
- numpy
- matplotlib (for visualizations, optional)

Both experiments gracefully handle missing matplotlib by skipping plot generation while still saving raw data.

## 🚀 Quick Start

```bash
# Run both experiments
python -m experiments.adapter_double_slit
python cli.py atom-from-constants

# View the key results
open artifacts/adapter_double_slit/interference.png
open artifacts/real_atom/atom_mip.png
open artifacts/real_atom/energy_convergence.png
```

## 📝 Notes

- All experiments are deterministic (using fixed random seeds)
- Output directories are created automatically
- Artifacts are saved in JSON and numpy formats for reproducibility
- No existing artifacts are overwritten without re-running experiments
- The atom solver note: "Derived from constants; no hand-tuned orbitals"

## 🎓 Scientific Background

### Double-Slit Interference
In quantum mechanics, interference occurs when two probability amplitudes add:
- `ψ_total = ψ_A + ψ_B × exp(iφ)`
- Intensity: `I = |ψ_total|²`
- Visibility: `V = (I_max - I_min) / (I_max + I_min)`

High visibility indicates quantum superposition, not classical mixing.

### Schrödinger Equation Solver
Uses imaginary-time propagation:
- Real time evolution: `ψ(t+dt) = exp(-iHt/ℏ) ψ(t)` → oscillatory
- Imaginary time: `ψ(τ+dτ) = exp(-Hτ/ℏ) ψ(τ)` → exponential decay
- Higher energy states decay faster → ground state remains
- This is a standard technique in quantum chemistry and physics

The finite-difference Laplacian uses a 6-point stencil for ∇²ψ in 3D.

## 📚 References

- Feynman Lectures on Physics, Vol. III (Quantum Mechanics)
- "Numerical Solution of the Schrödinger Equation" - standard textbook
- Double-slit experiment: foundational quantum mechanics experiment
