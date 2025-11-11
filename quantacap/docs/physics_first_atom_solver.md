# Physics-First Atom Solver

## Overview

The physics-first atom solver computes atomic wavefunctions by solving the time-independent Schrödinger equation from first principles. Unlike visualization tools that render pre-determined orbital shapes, this solver derives the electron density distribution directly from quantum mechanics.

## Key Features

- **No assumptions**: Starts from the Coulomb potential and fundamental constants
- **Imaginary-time evolution**: Robust numerical method for finding ground states
- **Physically accurate**: Results match theoretical predictions within numerical precision
- **Reproducible**: Complete descriptor JSON allows exact reconstruction

## Method

The solver uses imaginary-time propagation on a 3D Cartesian grid:

```
∂ψ/∂τ = (1/2) ∇²ψ - V(r) ψ
```

where:
- ψ is the wavefunction
- τ is imaginary time
- V(r) = -Z/√(r² + ε²) is the softened Coulomb potential
- Atomic units: ℏ = m_e = e = 1

This naturally converges to the lowest energy eigenstate (ground state).

## Usage

### Command Line

```bash
# Hydrogen atom (default)
python quantacap/src/quantacap/experiments/solve_atom_from_constants.py

# Helium+ ion (Z=2)
python quantacap/src/quantacap/experiments/solve_atom_from_constants.py \
  --Z 2.0 --L 8.0 --steps 800

# Quick test (low resolution)
python quantacap/src/quantacap/experiments/solve_atom_from_constants.py \
  --N 32 --steps 200
```

### CLI (when quantacap is installed)

```bash
quantacap solve-atom --Z 1.0 --N 64 --plot
```

### Python API

```python
from quantacap.experiments.solve_atom_from_constants import imaginary_time_solve

result = imaginary_time_solve(
    N=64,        # grid points per axis
    L=12.0,      # box size (atomic units)
    Z=1.0,       # nuclear charge
    dt=0.002,    # time step
    steps=600,   # iteration count
    softening=0.3
)

psi = result["psi"]          # wavefunction
density = result["density"]  # |ψ|²
energies = result["energies"] # convergence history
```

## Parameters

### Grid Parameters

- **N**: Number of grid points per axis
  - Typical: 32 (quick), 64 (standard), 128 (high-res)
  - Memory scales as N³

- **L**: Physical box size in atomic units
  - For hydrogen: L=12 is usually sufficient
  - For higher Z: can use smaller L (electron more tightly bound)

### Physical Parameters

- **Z**: Nuclear charge
  - Z=1: hydrogen
  - Z=2: helium+ (He⁺)
  - Z=3: lithium²⁺ (Li²⁺)

- **softening**: Nuclear softening radius
  - Prevents singularity at r=0
  - Default: 0.3 (reasonable for hydrogen)
  - Physical interpretation: finite nuclear size

### Numerical Parameters

- **steps**: Number of imaginary time steps
  - More steps → better convergence
  - Typical: 300-600
  - Check convergence plot to see if more needed

- **dt**: Imaginary time step size
  - Smaller → more stable but slower
  - Typical: 0.001-0.002
  - If energy diverges, reduce dt

## Output

### Files

All outputs saved to `artifacts/real_atom/`:

- `psi.npy`: 3D wavefunction array
- `density.npy`: 3D electron density
- `V.npy`: 3D potential field
- `atom_descriptor.json`: Complete specification
- `atom_mip.png`: Maximum intensity projection
- `slice_*.png`: Cross-sectional views
- `energy_convergence.png`: Energy vs iteration

### Descriptor JSON

The descriptor contains everything needed to reproduce the result:

```json
{
  "name": "REAL-ATOM-FROM-SCHRODINGER-V1",
  "grid": {
    "N": 64,
    "L": 12.0,
    "dx": 0.1935,
    "coords": {"x_min": -6.0, "x_max": 6.0}
  },
  "potential": {
    "type": "coulomb",
    "Z": 1.0,
    "softening": 0.3
  },
  "solver": {
    "method": "imaginary_time",
    "dt": 0.002,
    "steps": 600
  },
  "energies": [...],
  "notes": "Derived from constants; no hand-tuned orbitals."
}
```

## Validation

### Energy Comparison

For hydrogen in atomic units:

| Method | Energy (a.u.) |
|--------|--------------|
| Exact theory | -0.500000 |
| This solver (N=64) | -0.485 to -0.495 |
| Error | ~1-3% |

Error sources:
1. Finite grid spacing (dx)
2. Finite box size (L)
3. Nuclear softening (ε)

Higher N and larger L reduce error.

### Density Verification

The solver automatically ensures:
- Normalization: ∫|ψ|² dV = 1
- Spherical symmetry (for hydrogen)
- Exponential decay at large r

## Examples

### Compare Hydrogen vs Helium+

```python
# Hydrogen (Z=1)
h = imaginary_time_solve(N=48, L=12.0, Z=1.0, steps=400)
h_energy = h["energies"][-1][1]

# Helium+ (Z=2)
he = imaginary_time_solve(N=48, L=8.0, Z=2.0, steps=400, dt=0.001)
he_energy = he["energies"][-1][1]

# Scaling check: E scales as Z²
print(f"H energy: {h_energy:.4f} a.u.")
print(f"He+ energy: {he_energy:.4f} a.u.")
print(f"Ratio: {he_energy/h_energy:.2f} (expect ~4)")
```

### Extract Radial Distribution

```python
import numpy as np

result = imaginary_time_solve(N=64, L=12.0, Z=1.0, steps=600)
density = result["density"]
x = result["x"]
dx = result["dx"]

# Compute radial distribution
N = density.shape[0]
center = N // 2
radial_dist = []

for i in range(N // 2):
    # Sample density at distance r from center
    shell_density = density[center + i, center, center]
    r = i * dx
    # Weight by shell volume: 4πr²
    radial_dist.append(4 * np.pi * r**2 * shell_density)

# radial_dist[i] is now the probability density at radius r[i]
```

## Comparison with QCR

The physics-first solver complements the QCR (Quantum Coordinate Reconstruction) method:

| Feature | QCR | Physics-First |
|---------|-----|---------------|
| Input | Qubit measurements | Physical constants |
| Method | Reconstruction | Direct solution |
| Speed | Fast | Moderate |
| Accuracy | Synthetic | Exact (within numerics) |
| Use case | Quantum computer output | Ground truth validation |

Use QCR when you have quantum measurement data. Use physics-first when you want the "correct answer" from theory.

## Advanced: Different Potentials

To test exotic atoms or modified physics, edit the `coulomb_potential` function:

```python
def custom_potential(X, Y, Z, params):
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Example: screened Coulomb (Yukawa)
    V = -params["Z"] * np.exp(-r / params["lambda"]) / r
    
    return V
```

Then use `custom_potential` in place of `coulomb_potential` in the solver.

## Performance

Typical run times (single CPU core):

| N | Grid size | Time | Memory |
|---|-----------|------|--------|
| 32 | 32³ | ~10s | ~100 MB |
| 64 | 64³ | ~2 min | ~1 GB |
| 128 | 128³ | ~30 min | ~8 GB |

GPU acceleration not currently implemented but would help for large N.

## References

1. **Imaginary-time evolution**: Standard technique in quantum many-body physics
2. **Atomic units**: Simplify equations by setting ℏ = m_e = e = 1
3. **Finite-difference methods**: Chapter 9 of "Computational Physics" by M. Newman
4. **Hydrogen atom**: Theoretical ground state energy E = -0.5 a.u. = -13.6 eV

## See Also

- `qcr_atom_reconstruction.md` - Reconstruction from quantum measurements
- `exotic_atom_floquet.md` - Time-dependent driven atoms
- Example: `quantacap/examples/demo_solve_atom.py`
