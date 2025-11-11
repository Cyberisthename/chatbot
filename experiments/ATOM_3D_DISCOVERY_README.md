# 3D Atom Solver - Discovery Mode

## Overview

A physics-only, no-shortcuts 3D Schrödinger equation solver that discovers atomic structure from first principles.

**Key Features:**
- ✅ Solves from Schrödinger equation only (no pre-baked orbitals)
- ✅ Progressive resolution: 64³ → 128³ → 256³
- ✅ True 3D volume computation (not 2D projections)
- ✅ Auto-scaled time steps for numerical stability
- ✅ Exports all data for later analysis
- ✅ 4K-ready renders from real density volumes
- ✅ Multiple orthogonal views (XY, XZ, YZ projections)

## Running the Experiment

### Quick Start

```bash
# Run the full progressive solver
python -m experiments.solve_atom_3d_discovery

# Or use the CLI
python cli.py atom-3d-discovery
```

### What It Does

The solver:
1. **Stage 1 (64³)**: Starts from a random wavefunction and evolves it in imaginary time until it converges to the ground state
2. **Stage 2 (128³)**: Upsamples the result and refines it at higher resolution
3. **Stage 3 (256³)**: Final refinement at the highest resolution
4. **Renders**: Generates max-intensity projections from the 3D density volume

Each stage uses an automatically scaled time step to maintain numerical stability:
- dt(N=64) = 0.002
- dt(N=128) = 0.0005 (scaled by dx²)
- dt(N=256) = 0.000125 (scaled by dx²)

## Output Artifacts

All results are saved to `artifacts/real_atom_3d/`:

### 3D Volume Data
- `density_N64.npy`, `density_N128.npy`, `density_N256.npy` - Electron density |ψ|²
- `psi_N64.npy`, `psi_N128.npy`, `psi_N256.npy` - Wavefunction ψ
- `potential_N64.npy`, `potential_N128.npy`, `potential_N256.npy` - Potential V(r)

### Energy Convergence
- `energy_N64.json`, `energy_N128.json`, `energy_N256.json` - Energy vs step for each stage

### Renders (4K-ready)
- `atom_mip_xy.png` - Top view (max intensity projection along Z)
- `atom_mip_xz.png` - Side view (max intensity projection along Y)
- `atom_mip_yz.png` - Other side view (max intensity projection along X)
- `atom_spin.gif` - 360° rotation (if imageio is installed)

### Master Descriptor
- `atom3d_descriptor.json` - Complete metadata and configuration

## Physics Details

### Method: Imaginary-Time Propagation

We solve the time-independent Schrödinger equation:

```
Ĥψ = Eψ
```

where:
- Ĥ = -½∇² + V(r) (Hamiltonian in atomic units)
- V(r) = -Z/√(r² + ε²) (Coulomb potential with softening ε)

By evolving in imaginary time τ = it:

```
∂ψ/∂τ = ½∇²ψ - V(r)ψ
```

and normalizing at each step, the wavefunction automatically relaxes to the ground state.

### Numerical Stability

The finite-difference Laplacian scales as 1/dx², so we auto-scale the time step:

```
dt_scaled = dt₀ × (dx / dx₀)²
```

This maintains the CFL stability condition across all resolutions.

### Configuration

Default parameters (can be modified in `CONFIG` dict):

```python
CONFIG = {
    "N_stages": [64, 128, 256],     # grid resolutions
    "box": 12.0,                    # physical box size (atomic units)
    "Z": 1.0,                       # nuclear charge (1 = hydrogen)
    "softening": 0.3,               # nuclear softening parameter
    "dt": 0.002,                    # base time step
    "steps_per_stage": 400,         # evolution steps per stage
    "save_every": 50,               # save energy every N steps
    "centers": [[0.0, 0.0, 0.0]],   # nuclear positions
    "seed": 424242,                 # random seed for reproducibility
}
```

### Multi-Center Potentials

To solve H₂ or other multi-center systems, just add more centers:

```python
CONFIG["centers"] = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]  # H₂-like
```

## Expected Results

For hydrogen (Z=1):
- **Energy**: Ground state energy should converge to ~-0.39 to -0.42 a.u. (theoretical = -0.5 a.u.)
- **Density**: Spherically symmetric cloud centered at nucleus
- **Size**: ~1-2 Bohr radii (a₀)

The energy is slightly higher than the exact value due to:
- Finite grid resolution
- Nuclear softening (avoids singularity)
- Limited evolution time

## Facts-Only Mode

`FACTS_ONLY = True` means:
- ❌ No pre-loaded hydrogen orbitals
- ❌ No Gaussian smoothing
- ❌ No artificial symmetrization
- ❌ No spherical harmonics
- ✅ Only: Laplacian + potential + evolution + normalization

What you see is what the math gives you, nothing more.

## Memory Requirements

Approximate memory usage:
- **N=64**: ~10 MB per array × 3 arrays = ~30 MB
- **N=128**: ~130 MB per array × 3 arrays = ~400 MB
- **N=256**: ~1 GB per array × 3 arrays = ~3 GB

If you hit a `MemoryError` at N=256, the solver will save what it computed at N=128 and stop gracefully.

## Performance

Typical run times (on a modern laptop):
- **N=64**: ~30 seconds
- **N=128**: ~2-3 minutes
- **N=256**: ~10-15 minutes

## Viewing Results

After running, open:

```bash
# View the 3D atom from different angles
artifacts/real_atom_3d/atom_mip_xy.png  # top view
artifacts/real_atom_3d/atom_mip_xz.png  # side view
artifacts/real_atom_3d/atom_mip_yz.png  # other side

# View the metadata
artifacts/real_atom_3d/atom3d_descriptor.json
```

## Extending the Experiment

### Add More Resolutions

```python
CONFIG["N_stages"] = [64, 128, 256, 512]  # add 512³ if you have RAM
```

### Change the Nucleus

```python
CONFIG["Z"] = 2.0  # Helium (but you'd need to add electron-electron repulsion)
```

### Solve for Excited States

After finding the ground state, orthogonalize and evolve again:

```python
# (requires additional code to implement excited state solver)
```

### Export to Other Tools

All volumes are standard numpy arrays:

```python
import numpy as np
density = np.load("artifacts/real_atom_3d/density_N256.npy")
# Use density with your favorite 3D visualization tool
```

## Troubleshooting

### Energy diverges at high resolution

Reduce the time step:
```python
CONFIG["dt"] = 0.001  # smaller dt = more stable
```

### Out of memory

Reduce max resolution:
```python
CONFIG["N_stages"] = [64, 128]  # skip 256³
```

### Energy not converging

Increase evolution time:
```python
CONFIG["steps_per_stage"] = 800  # more steps = better convergence
```

## References

- Imaginary-time propagation: Beck et al., Rev. Mod. Phys. 72, 1041 (2000)
- Finite-difference Laplacian: Numerical recipes in scientific computing
- CFL stability condition: Courant, Friedrichs, Lewy (1928)

---

**Questions?** This experiment is designed to be self-contained and reproducible. All physics is explicit in the code.
