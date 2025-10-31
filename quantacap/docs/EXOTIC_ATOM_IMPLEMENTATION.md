# Exotic Atom Floquet Implementation Summary

## What Was Added

This implementation adds a new quantum experiment to the Quantacap project that simulates a Floquet-driven system with long-range interactions and visualizes the results as a synthetic "atom" density.

## Files Added

### 1. Main Experiment Module
**Location**: `quantacap/src/quantacap/experiments/exotic_atom_floquet.py`

This module implements:
- Pauli matrix operations and Kronecker products for multi-qubit systems
- Long-range ZZ Hamiltonian construction with power-law decay
- Time evolution using matrix exponentiation
- Entanglement entropy calculation via partial trace
- Z-expectation measurements for all qubits
- 2D density rendering from qubit probabilities
- PNG and animated GIF generation

### 2. CLI Integration
**Location**: `quantacap/src/quantacap/cli.py` (modified)

Added:
- Import for `run_exotic_atom_floquet` function
- `_exotic_atom_cmd()` handler function
- CLI argument parser for the `exotic-atom` subcommand
- Full set of configurable parameters

### 3. Documentation
**Location**: `quantacap/docs/exotic_atom_floquet.md`

Comprehensive documentation including:
- Physics background
- Usage examples
- Parameter descriptions
- Output format specification
- Performance notes

### 4. Example Script
**Location**: `quantacap/examples/exotic_atom_example.sh`

Shell script demonstrating various usage patterns with different parameter combinations.

### 5. Implementation Notes
**Location**: `quantacap/docs/EXOTIC_ATOM_IMPLEMENTATION.md` (this file)

## CLI Usage

```bash
# Set PYTHONPATH to include quantacap source
export PYTHONPATH="/path/to/quantacap/src:/path/to/quantacap:${PYTHONPATH}"

# Run with defaults
python -m quantacap.cli exotic-atom

# Run with custom parameters
python -m quantacap.cli exotic-atom \
  --N 8 \
  --steps 120 \
  --alpha 1.3 \
  --drive-amp 1.0 \
  --drive-freq 2.0
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--N` | int | 8 | Number of qubits in the system |
| `--steps` | int | 80 | Number of time evolution steps |
| `--dt` | float | 0.05 | Time step size for evolution |
| `--drive-amp` | float | 1.0 | Amplitude of periodic drive |
| `--drive-freq` | float | 2.0 | Frequency of periodic drive |
| `--J-nn` | float | 1.0 | Nearest-neighbor coupling strength |
| `--J-lr` | float | 0.5 | Long-range coupling strength |
| `--alpha` | float | 1.5 | Long-range interaction decay exponent |
| `--seed` | int | 424242 | Random seed for reproducibility |
| `--no-gif` | flag | False | Skip GIF generation (only PNG) |
| `--out` | string | artifacts/exotic_atom_floquet.json | Output path for JSON results |

## Output Files

1. **JSON file** (default: `artifacts/exotic_atom_floquet.json`):
   - Complete experiment parameters
   - Time series data (entanglement, energy, drive values)
   - Final qubit measurements
   - Artifact file paths

2. **PNG image** (`artifacts/exotic_atom_density.png`):
   - 2D visualization of final quantum state
   - Color-mapped density plot
   - 160 DPI resolution

3. **Animated GIF** (`artifacts/exotic_atom_evolution.gif`):
   - Time evolution of the atom density
   - Frame duration: 120ms
   - Looping animation

## Python API

```python
from quantacap.experiments.exotic_atom_floquet import run_exotic_atom_floquet

result = run_exotic_atom_floquet(
    N=8,                    # 8 qubits
    steps=80,               # 80 evolution steps
    dt=0.05,                # time step
    drive_amp=1.0,          # drive amplitude
    drive_freq=2.0,         # drive frequency
    J_nn=1.0,               # nearest-neighbor coupling
    J_lr=0.5,               # long-range coupling
    alpha=1.5,              # decay exponent
    seed=424242,            # random seed
    make_gif=True,          # generate animated GIF
    out_json="artifacts/exotic_atom_floquet.json"
)

# result is a dictionary containing all experiment data
```

## Physics Details

### Hamiltonian Structure

The total time-dependent Hamiltonian is:
```
H(t) = H_static + H_drive(t)
```

Where:
- **H_static** includes:
  - Nearest-neighbor: `J_nn * Σ_i Z_i Z_{i+1}`
  - Long-range: `Σ_{i<j} [J_lr * r_ij / |i-j|^α] Z_i Z_j`
  - `r_ij` are random coefficients in [-1, 1]

- **H_drive(t)** is:
  - `A * cos(ω*t) * Σ_i X_i`
  - Periodic transverse field drive

### Evolution Method

Time evolution uses exact matrix exponentiation:
```
|ψ(t+dt)⟩ = exp(-i H(t) dt) |ψ(t)⟩
```

Implemented via eigenvalue decomposition for numerical stability.

### Measurements

- **Entanglement Entropy**: Von Neumann entropy of reduced density matrix
  - Bipartition at middle of qubit chain
  - `S = -Tr(ρ_A log₂ ρ_A)`

- **Energy**: Expectation value `⟨ψ|H|ψ⟩`

- **Local Z-values**: Single-qubit magnetization `⟨Z_i⟩`

### Visualization Mapping

Qubits arranged on a circle:
- Qubit k at angle `θ_k = 2πk/N`
- Position: `(x, y) = r(cos θ_k, sin θ_k)` with r=0.55
- Density contribution: Gaussian blob with weight `p₁ = (1 - ⟨Z_k⟩)/2`
- Blob width: σ = 0.30

## Performance Characteristics

| N qubits | State Dim | Memory (GB) | Approx. Time (80 steps) |
|----------|-----------|-------------|-------------------------|
| 5 | 32 | ~0.001 | < 1 second |
| 6 | 64 | ~0.001 | < 2 seconds |
| 7 | 128 | ~0.001 | ~5 seconds |
| 8 | 256 | ~0.002 | ~15 seconds |
| 9 | 512 | ~0.004 | ~1 minute |
| 10 | 1024 | ~0.008 | ~5 minutes |

Note: Times are approximate and depend on CPU. Memory is for complex128.

## Dependencies

Required:
- `numpy` - Core array operations and linear algebra

Optional (for visualization):
- `matplotlib` - PNG generation
- `PIL` (Pillow) - GIF animation

The experiment will run without optional dependencies but won't generate visualizations.

## Integration with Quantacap

This experiment follows the same patterns as other Quantacap experiments:
- Consistent CLI interface
- JSON output format
- Artifact generation in `artifacts/` directory
- Optional visualization support
- Configurable via command-line arguments
- Importable as Python function
- Reproducible via seed parameter

## Future Enhancements

Possible extensions:
- 3D visualization of multi-qubit correlations
- Support for different lattice geometries
- Additional measurement observables (correlators, etc.)
- Quantum circuit decomposition of the evolution
- Comparison with classical Ising model
- Phase diagram exploration in parameter space
