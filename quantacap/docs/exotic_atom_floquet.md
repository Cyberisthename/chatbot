# Exotic Atom Floquet Experiment

## Overview

The exotic atom Floquet experiment simulates a periodically-driven quantum system with long-range interactions, visualizing the quantum state evolution as a synthetic "atom" density.

## Features

- **Floquet Dynamics**: Time-periodic driving with adjustable amplitude and frequency
- **Long-Range Hamiltonian**: Combines nearest-neighbor and power-law decaying long-range ZZ interactions
- **Quantum Measurements**: Tracks entanglement entropy and energy evolution
- **Visualization**: Generates PNG snapshot and animated GIF of the synthetic atom density

## Physics

The Hamiltonian consists of:

1. **Static part**: 
   - Nearest-neighbor: `H_nn = J_nn * Σ Z_i Z_{i+1}`
   - Long-range: `H_lr = Σ_{i<j} (J_lr * rand / |i-j|^α) Z_i Z_j`

2. **Drive term**: `H_drive(t) = A * cos(ω*t) * Σ X_i`

The system evolves under `H(t) = H_static + H_drive(t)` with time step `dt`.

## Usage

### Command Line

```bash
python -m quantacap.cli exotic-atom [OPTIONS]
```

### Options

- `--N INT`: Number of qubits (default: 8)
- `--steps INT`: Number of time evolution steps (default: 80)
- `--dt FLOAT`: Time step size (default: 0.05)
- `--drive-amp FLOAT`: Drive amplitude (default: 1.0)
- `--drive-freq FLOAT`: Drive frequency (default: 2.0)
- `--J-nn FLOAT`: Nearest-neighbor coupling strength (default: 1.0)
- `--J-lr FLOAT`: Long-range coupling strength (default: 0.5)
- `--alpha FLOAT`: Long-range decay exponent (default: 1.5)
- `--seed INT`: Random seed for reproducibility (default: 424242)
- `--no-gif`: Skip GIF generation (only create PNG)
- `--out PATH`: Output JSON path (default: artifacts/exotic_atom_floquet.json)

### Examples

**Basic run with defaults:**
```bash
python -m quantacap.cli exotic-atom
```

**Larger system with custom parameters:**
```bash
python -m quantacap.cli exotic-atom --N 8 --steps 120 --alpha 1.3
```

**Adjust drive parameters:**
```bash
python -m quantacap.cli exotic-atom \
  --drive-amp 1.5 \
  --drive-freq 3.0 \
  --alpha 1.2
```

**Skip GIF generation for faster execution:**
```bash
python -m quantacap.cli exotic-atom --N 7 --steps 100 --no-gif
```

### Python API

You can also import and run the experiment directly:

```python
from quantacap.experiments.exotic_atom_floquet import run_exotic_atom_floquet

result = run_exotic_atom_floquet(
    N=8,
    steps=80,
    dt=0.05,
    drive_amp=1.0,
    drive_freq=2.0,
    J_nn=1.0,
    J_lr=0.5,
    alpha=1.5,
    seed=424242,
    make_gif=True,
    out_json="artifacts/exotic_atom_floquet.json"
)

print(result)
```

## Output

The experiment generates three artifacts:

1. **JSON file** (`exotic_atom_floquet.json`): Contains:
   - Experiment parameters
   - Time series of entanglement entropy
   - Time series of system energy
   - Drive amplitude at each step
   - Final Z-expectation values for each qubit

2. **PNG image** (`exotic_atom_density.png`): 
   - 2D density plot of the final quantum state
   - Qubits arranged on a circle
   - Density proportional to |1⟩ probability

3. **Animated GIF** (`exotic_atom_evolution.gif`):
   - Time evolution of the atom density
   - Shows how the quantum state evolves under the Floquet drive

## Interpretation

The "synthetic atom" visualization maps the quantum state onto a 2D density:
- Each qubit is placed at a position on a circle
- The probability of measuring |1⟩ determines the local density
- Gaussian blobs centered at each qubit position create a smooth density field
- The visualization reveals patterns and dynamics in the multi-qubit entangled state

## Performance Notes

- **Computational complexity**: Scales exponentially with N (state dimension = 2^N)
- **Recommended range**: N ≤ 10 qubits for reasonable compute times
- **Memory usage**: Approximately 2^N complex128 values for the state vector
- **GIF generation**: Can be slow for large step counts; use `--no-gif` to disable

## Dependencies

- `numpy`: Required for quantum simulation
- `matplotlib`: Required for PNG generation
- `PIL` (Pillow): Required for GIF animation
