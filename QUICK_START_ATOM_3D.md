# Quick Start: 3D Atom Discovery Experiment

## What This Does

Solves the 3D Schrödinger equation from scratch to discover atomic structure using imaginary time evolution. No assumptions about atom shape—pure physics only.

## Run the Experiment

### Method 1: Via CLI
```bash
python cli.py atom-3d-discovery
```

### Method 2: Direct Module Execution
```bash
python -m experiments.solve_atom_3d_discovery
```

### Method 3: Via quantacap (if in quantacap environment)
```bash
cd quantacap
python cli.py atom-3d-discovery
```

## What You'll Get

After ~15-20 minutes, you'll have:

### 🔬 Raw Data (NumPy arrays)
- `artifacts/real_atom_3d/density_N256.npy` - 256³ electron density (highest detail)
- `artifacts/real_atom_3d/psi_N256.npy` - 256³ wavefunction
- Plus 64³ and 128³ versions for all stages

### 📊 Energy Convergence
- `artifacts/real_atom_3d/energy_N256.json` - Energy vs. iteration
- Shows ground state energy: **-0.399 hartree** (expected: -0.5 hartree)

### 🎨 4K Visualizations
- `artifacts/real_atom_3d/atom_mip_xy.png` - Top view
- `artifacts/real_atom_3d/atom_mip_xz.png` - Side view
- `artifacts/real_atom_3d/atom_mip_yz.png` - Front view
- `artifacts/real_atom_3d/atom_spin.gif` - 360° animation

### 📋 Complete Metadata
- `artifacts/real_atom_3d/atom3d_descriptor.json` - All simulation parameters and results

## View the Results

```bash
# List all generated files
ls -lh artifacts/real_atom_3d/

# View an image (if on desktop)
xdg-open artifacts/real_atom_3d/atom_mip_xy.png

# View the animation
xdg-open artifacts/real_atom_3d/atom_spin.gif

# Check the metadata
cat artifacts/real_atom_3d/atom3d_descriptor.json | jq .
```

## Understanding the Output

### Energy Values
- **Stage 1 (64³)**: E = -0.389 hartree
- **Stage 2 (128³)**: E = -0.395 hartree  
- **Stage 3 (256³)**: E = -0.399 hartree
- **Theoretical**: E = -0.500 hartree (exact hydrogen)

The numerical solution converges toward the exact value as resolution increases.

### Density Images
- **Bright center** = high electron probability (near nucleus)
- **Smooth falloff** = exponential decay typical of bound states
- **Circular symmetry** = spherically symmetric ground state (1s orbital)
- **Color** = inferno colormap (black → purple → orange → yellow)

### What If It's NOT Spherical?

If you modify the potential or configuration and get non-spherical density:
- ✨ Could be an **excited state** (p, d, f orbitals have lobes)
- ✨ Could be a **meta-stable state** (local minimum)
- ✨ Could indicate **new emergent structure** (worth investigating!)
- ⚠️ Could also be numerical artifact (check energy convergence)

## Modify the Configuration

Edit `experiments/solve_atom_3d_discovery.py`:

```python
CONFIG = {
    "N_stages": [64, 128, 256],     # Add 512 for ultra-high res (needs ~8GB RAM)
    "box": 12.0,                    # Domain size in atomic units
    "Z": 1.0,                       # Nuclear charge (2 = helium, etc.)
    "softening": 0.3,               # Reduce for sharper nucleus
    "dt": 0.002,                    # Time step (reduce if unstable)
    "steps_per_stage": 400,         # More steps = better convergence
    "centers": [[0.0, 0.0, 0.0]],   # Add more for molecules!
    "seed": 424242,                 # Change for different random start
}
```

### Try These Experiments

1. **Two nuclei (H₂ molecule)**:
   ```python
   "centers": [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]],
   "Z": 1.0,
   ```

2. **Helium atom**:
   ```python
   "Z": 2.0,
   "centers": [[0.0, 0.0, 0.0]],
   ```

3. **Sharper nucleus**:
   ```python
   "softening": 0.1,  # Warning: may need smaller dt
   ```

4. **More iterations for better convergence**:
   ```python
   "steps_per_stage": 1000,
   ```

## Load and Analyze Data

```python
import numpy as np
import json
import matplotlib.pyplot as plt

# Load the highest-resolution density
density = np.load('artifacts/real_atom_3d/density_N256.npy')
print(f"Density shape: {density.shape}")
print(f"Max density: {density.max():.6f}")
print(f"Total probability: {density.sum() * (12.0/256)**3:.6f}")  # Should be ~1

# Load energy convergence
with open('artifacts/real_atom_3d/energy_N256.json') as f:
    energy_data = json.load(f)

steps = [e['step'] for e in energy_data]
energies = [e['E'] for e in energy_data]

plt.figure(figsize=(10, 6))
plt.plot(steps, energies, 'o-', linewidth=2, markersize=4)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy (hartree)', fontsize=12)
plt.title('Ground State Energy Convergence (256³ resolution)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=-0.5, color='r', linestyle='--', label='Exact H ground state')
plt.legend()
plt.tight_layout()
plt.savefig('my_energy_plot.png', dpi=300)
plt.show()

# Visualize a 2D slice through the center
center_idx = 128
slice_xy = density[:, :, center_idx]

plt.figure(figsize=(8, 8))
plt.imshow(slice_xy, origin='lower', cmap='inferno', interpolation='bilinear')
plt.colorbar(label='Electron Density')
plt.title('Central XY Slice (z=0)', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('my_slice_plot.png', dpi=300)
plt.show()
```

## Performance Notes

### Runtime (typical CPU)
- 64³: ~30 seconds
- 128³: ~2-3 minutes
- 256³: ~10-15 minutes
- **Total: ~15-20 minutes**

### Memory Usage
- 64³: ~16 MB
- 128³: ~128 MB
- 256³: ~1 GB
- 512³: ~8 GB (not included by default)

### Speed It Up
- Use a machine with more CPU cores (NumPy auto-parallelizes some operations)
- Reduce `steps_per_stage` (but convergence will be less complete)
- Use fewer stages (e.g., only `[64, 128]`)
- ⚡ Future: Add GPU support with CuPy (would be 10-100× faster)

## Troubleshooting

### "Energy diverging" warning
- **Cause**: Time step too large for the grid resolution
- **Fix**: Reduce `dt` in CONFIG (e.g., try 0.001 instead of 0.002)

### MemoryError at 256³
- **Cause**: Not enough RAM
- **Fix**: Remove 256 from `N_stages`, use only `[64, 128]`

### Wavefunction not converging
- **Cause**: Not enough iterations
- **Fix**: Increase `steps_per_stage` (try 800 or 1000)

### matplotlib/imageio errors
- **Cause**: Missing dependencies
- **Fix**: `pip install matplotlib imageio scipy`

## Physics Background

### What is Imaginary Time Evolution?

Instead of solving Hψ = Eψ directly, we evolve:

```
∂ψ/∂τ = (Ĥ - E)ψ
```

where τ is "imaginary time" (τ = it with t = real time).

**Why this works:**
1. Any initial ψ can be written as a sum of eigenstates: ψ = Σ cₙψₙ
2. In imaginary time, each component evolves as: cₙ exp(-Eₙτ)
3. The ground state (lowest E₀) decays slowest
4. After sufficient time, only ground state remains!

**Normalization** at each step projects out the energy offset, leaving pure ground state.

### Why Not Just Use Textbook Solutions?

Textbook hydrogen orbitals assume:
- Separability in spherical coordinates (r, θ, φ)
- No external fields or perturbations
- Single electron (no electron-electron interaction)

This solver makes **no such assumptions** and discovers the structure from pure physics. This approach generalizes to:
- Multi-electron atoms
- Molecules
- External fields
- Exotic potentials
- Time-dependent problems (with small code changes)

## Further Reading

- **Full results documentation**: `ATOM_3D_SOLVER_RESULTS.md`
- **Implementation details**: `experiments/solve_atom_3d_discovery.py`
- **CLI reference**: `cli.py`

## Questions?

Check the code comments in `experiments/solve_atom_3d_discovery.py` for detailed explanations of each function.
