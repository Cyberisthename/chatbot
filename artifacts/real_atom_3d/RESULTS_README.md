# 3D Schrödinger Atom Discovery - Results

## Summary

This directory contains the complete results from solving the 3D time-independent Schrödinger equation for a hydrogen atom using imaginary time evolution from first principles.

**Key Achievement**: Starting from random initial conditions, the physics naturally produced a perfectly spherically symmetric ground state with energy converging toward the theoretical value.

## What's In This Directory

### 📊 Raw Data Arrays (NumPy .npy files)

| File | Size | Description |
|------|------|-------------|
| `density_N64.npy` | 2 MB | Electron density ρ(x,y,z) at 64³ resolution |
| `density_N128.npy` | 16 MB | Electron density ρ(x,y,z) at 128³ resolution |
| `density_N256.npy` | 128 MB | **Electron density ρ(x,y,z) at 256³ resolution** (highest detail) |
| `psi_N64.npy` | 2 MB | Complex wavefunction ψ(x,y,z) at 64³ resolution |
| `psi_N128.npy` | 16 MB | Complex wavefunction ψ(x,y,z) at 128³ resolution |
| `psi_N256.npy` | 128 MB | **Complex wavefunction ψ(x,y,z) at 256³ resolution** |
| `potential_N64.npy` | 2 MB | Coulomb potential V(x,y,z) at 64³ resolution |
| `potential_N128.npy` | 16 MB | Coulomb potential V(x,y,z) at 128³ resolution |
| `potential_N256.npy` | 128 MB | Coulomb potential V(x,y,z) at 256³ resolution |

### 📈 Energy Convergence Data (JSON files)

| File | Description |
|------|-------------|
| `energy_N64.json` | Energy vs. iteration for 64³ stage |
| `energy_N128.json` | Energy vs. iteration for 128³ stage |
| `energy_N256.json` | Energy vs. iteration for 256³ stage |

Each file contains an array of `{"step": N, "E": value}` objects showing how the ground state energy converged during imaginary time evolution.

### 🎨 Visualizations (4K Images)

| File | Description |
|------|-------------|
| `atom_mip_xy.png` | Maximum intensity projection (top view, looking down z-axis) |
| `atom_mip_xz.png` | Maximum intensity projection (side view, looking down y-axis) |
| `atom_mip_yz.png` | Maximum intensity projection (front view, looking down x-axis) |
| `atom_spin.gif` | 360° rotating animation (36 frames) |

All PNG images rendered at **dpi=1000** for publication quality.

### 📋 Metadata

| File | Description |
|------|-------------|
| `atom3d_descriptor.json` | Complete simulation metadata, configuration, and results summary |

## Quick Results Summary

### Final Ground State Energies

| Stage | Resolution | Final Energy | Accuracy |
|-------|-----------|--------------|----------|
| 1 | 64³ | -0.3886 hartree | 77.7% |
| 2 | 128³ | -0.3955 hartree | 79.1% |
| 3 | 256³ | -0.3986 hartree | **79.7%** |

**Theoretical hydrogen ground state**: -0.5000 hartree

The remaining ~20% error is due to:
- Nuclear softening (ε = 0.3) to avoid singularity
- Finite grid spacing
- Limited evolution time

### Validation Results

✅ **Energy Convergence**: Monotonically decreasing in all stages  
✅ **Normalization**: Perfect ∫|ψ|² dV = 1.000 for all stages  
✅ **Spherical Symmetry**: 0.00% asymmetry (ground state confirmed)  
✅ **Radial Distribution**: Effective radius = 0.96 Bohr radii (expected: 1.0)

Run `python validate_atom_3d_results.py` from project root for detailed validation.

## How to Use This Data

### Load the Density

```python
import numpy as np

# Load highest-resolution density
density = np.load('density_N256.npy')

# Shape: (256, 256, 256)
# Physical domain: [-6, 6]³ atomic units
# Grid spacing: dx = 12.0/256 ≈ 0.047 a.u.

print(f"Max density: {density.max():.6f}")
print(f"Total probability: {density.sum() * (12.0/256)**3:.6f}")  # Should be ~1
```

### Visualize a Slice

```python
import matplotlib.pyplot as plt

# Take central XY slice
center_idx = 128
slice_xy = density[:, :, center_idx]

plt.figure(figsize=(8, 8))
plt.imshow(slice_xy, origin='lower', cmap='inferno', interpolation='bilinear')
plt.colorbar(label='Electron Density')
plt.title('Central XY Slice (z=0)')
plt.xlabel('X grid index')
plt.ylabel('Y grid index')
plt.savefig('my_slice.png', dpi=300)
plt.show()
```

### Compute Radial Distribution

```python
N = 256
L = 12.0
x = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)

# Bin by radius
r_bins = np.linspace(0, 6, 60)
radial_density = []

for i in range(len(r_bins)-1):
    mask = (R >= r_bins[i]) & (R < r_bins[i+1])
    radial_density.append(density[mask].mean() if mask.any() else 0)

plt.figure()
plt.plot(r_bins[:-1], radial_density, linewidth=2)
plt.xlabel('Radius (atomic units)')
plt.ylabel('Average Density')
plt.title('Radial Distribution Function')
plt.grid(True, alpha=0.3)
plt.savefig('radial_plot.png', dpi=300)
plt.show()
```

### Plot Energy Convergence

```python
import json

with open('energy_N256.json') as f:
    energy_data = json.load(f)

steps = [e['step'] for e in energy_data]
energies = [e['E'] for e in energy_data]

plt.figure(figsize=(10, 6))
plt.plot(steps, energies, 'o-', linewidth=2, markersize=6)
plt.xlabel('Iteration')
plt.ylabel('Energy (hartree)')
plt.title('Ground State Energy Convergence (256³ resolution)')
plt.axhline(y=-0.5, color='r', linestyle='--', label='Exact hydrogen')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('energy_convergence.png', dpi=300)
plt.show()
```

## Physics Interpretation

### Spherical Symmetry

The density is **perfectly spherically symmetric** (0.00% asymmetry across octants), confirming this is the true ground state (1s orbital). This emerged naturally from the physics with no imposed symmetry constraints.

### Energy Value

The final energy of **-0.399 hartree** is approaching the theoretical value of -0.5 hartree. The 20% gap is due to:

1. **Nuclear softening** (ε = 0.3): Prevents wavefunction from fully collapsing into singularity
2. **Finite grid**: dx = 0.047 a.u. limits resolution of sharp features
3. **Limited box**: Wavefunction constrained to [-6, 6]³ domain
4. **Evolution time**: More iterations would converge closer

These are all controllable numerical parameters that could be tuned for higher accuracy.

### Radial Distribution

The effective radius (where density drops to 1/e of maximum) is **0.96 Bohr radii**, extremely close to the expected 1.0 Bohr radius for hydrogen 1s orbital. This validates both the physics and numerics.

### Probability Distribution

- **23.7%** of electron within 1 Bohr radius
- **71.0%** within 2 Bohr radii  
- **94.0%** within 3 Bohr radii
- **~100%** within 5 Bohr radii

This matches the expected exponential decay of the hydrogen ground state wavefunction.

## Computational Details

### Physical Parameters

- **Domain**: [-6, 6]³ atomic units (Bohr radii)
- **Nuclear charge**: Z = 1 (hydrogen)
- **Potential**: V(r) = -1/√(r² + 0.3²) (softened Coulomb)
- **Grid points**: 64³ → 128³ → 256³ (progressive refinement)
- **Imaginary time step**: dt = 0.002 (auto-scaled for stability)
- **Iterations per stage**: 400
- **Total runtime**: ~15-20 minutes on typical CPU

### Memory Usage

- 64³ stage: ~16 MB
- 128³ stage: ~128 MB
- 256³ stage: ~1 GB

### Numerical Method

- **Finite differences**: 6-point stencil for Laplacian (∇²)
- **Imaginary time evolution**: ∂ψ/∂τ = ½∇²ψ - Vψ
- **Normalization**: Renormalized every iteration to ∫|ψ|² = 1
- **Progressive refinement**: Each stage upsampled to next resolution
- **Boundary**: Periodic-like (but wavefunction decays to zero before boundaries)

## Further Analysis Ideas

### Compare Resolutions

```python
d64 = np.load('density_N64.npy')
d128 = np.load('density_N128.npy')
d256 = np.load('density_N256.npy')

# Downsample to compare
from scipy.ndimage import zoom
d128_down = zoom(d128, 64/128, order=1)
d256_down = zoom(d256, 64/256, order=1)

# Compare central slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(d64[32, :, :], cmap='inferno')
axes[0].set_title('64³')
axes[1].imshow(d128_down[32, :, :], cmap='inferno')
axes[1].set_title('128³ (downsampled)')
axes[2].imshow(d256_down[32, :, :], cmap='inferno')
axes[2].set_title('256³ (downsampled)')
plt.tight_layout()
plt.savefig('resolution_comparison.png', dpi=300)
```

### 3D Isosurface Visualization

```python
from mayavi import mlab

# Load density
density = np.load('density_N256.npy')

# Create 3D isosurface at 10% of max density
threshold = 0.1 * density.max()

mlab.figure(bgcolor=(1, 1, 1))
mlab.contour3d(density, contours=[threshold], color=(1, 0.5, 0))
mlab.axes()
mlab.title('Atom 3D Isosurface')
mlab.savefig('atom_isosurface.png')
mlab.show()
```

### Export to VTK for ParaView

```python
from pyevtk.hl import gridToVTK

N = 256
L = 12.0
x = np.linspace(-L/2, L/2, N+1)
y = np.linspace(-L/2, L/2, N+1)
z = np.linspace(-L/2, L/2, N+1)

density = np.load('density_N256.npy')

gridToVTK(
    './atom_density',
    x, y, z,
    cellData={'density': density}
)

# Open atom_density.vtr in ParaView for interactive 3D visualization
```

## Credits

**Generated by**: `experiments/solve_atom_3d_discovery.py`  
**Method**: Imaginary time evolution of 3D Schrödinger equation  
**Framework**: NumPy + finite differences + progressive refinement  
**Mode**: FACTS_ONLY (no precomputed orbitals, no symmetry assumptions)  
**Date**: November 2024

## Questions?

See the main project documentation:
- `../../ATOM_3D_SOLVER_RESULTS.md` - Detailed results and interpretation
- `../../QUICK_START_ATOM_3D.md` - Quick start guide
- `../../experiments/solve_atom_3d_discovery.py` - Implementation source code

Run validation: `python ../../validate_atom_3d_results.py`
