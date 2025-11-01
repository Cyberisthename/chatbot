# QCR Atom Reconstruction

**QCR** = **Quantum Coordinate Reconstruction**

This module reconstructs a full 3D atomic density from qubit-like probability measurements (or synthetic data). It simulates an atom's electron cloud structure by iteratively refining a 3D density field until convergence, creating reproducible artifacts that any AI can use to rebuild the same atom.

## Overview

The QCR atom reconstruction process:

1. **Samples a full 3D cube** (x, y, z) in coordinate space
2. **Generates/loads qubit weights** - either from real quantum measurements or synthetic data
3. **Iteratively refines density** - converges through repeated "measurement-like" observations
4. **Exports comprehensive artifacts**:
   - 3D density field (`.npy`)
   - Isosurface mask (`.npy`)
   - Atom constant JSON (reproducible spec)
   - 2D slice images (`.png`)
   - Fly-through animation (`.gif`)
   - Convergence plot (`.png`)

## Usage

### Command Line

```bash
# Basic usage with default parameters
python -m quantacap.experiments.qcr_atom_reconstruct

# Custom parameters
python -m quantacap.experiments.qcr_atom_reconstruct \
  --N 72 \
  --iters 120 \
  --iso 0.35 \
  --n-qubits 8 \
  --seed 424242
```

### Parameters

- `--N` (int, default=64): Grid size per axis (N×N×N voxels)
- `--R` (float, default=1.0): Spatial radius in atomic units
- `--iters` (int, default=100): Maximum reconstruction iterations
- `--iso` (float, default=0.35): Isosurface threshold for surface extraction
- `--n-qubits` (int, default=8): Number of synthetic qubits to simulate
- `--seed` (int, default=424242): Random seed for reproducibility

### Python API

```python
from quantacap.experiments.qcr_atom_reconstruct import run_qcr_atom

# Run with custom parameters
summary = run_qcr_atom(
    N=72,           # 72×72×72 grid
    R=1.0,          # ±1.0 atomic units
    iters=120,      # up to 120 iterations
    iso=0.35,       # 35% density threshold
    n_qubits=8,     # 8 synthetic qubits
    seed=424242     # reproducible
)

print(summary)  # JSON summary of all artifacts
```

## Output Artifacts

All artifacts are saved to `artifacts/qcr/`:

### Data Files

- **`atom_density.npy`**: Full 3D density field (N×N×N float32 array)
  - Raw density values normalized to [0, 1]
  - Can be re-visualized or fed to 3D tools
  
- **`atom_isomask.npy`**: Boolean isosurface mask (N×N×N bool array)
  - `True` where density ≥ iso threshold
  - Ready for marching cubes or mesh extraction

- **`atom_constant.json`**: Reproducible atom specification
  - Grid parameters (R, N)
  - Isosurface threshold
  - Qubit weights (z-expectations)
  - Convergence history
  - **Any AI can rebuild the exact same atom from this file**

### Visualizations

- **`slice_0.png` ... `slice_4.png`**: 2D slices through the atom
  - Shows density at different z-levels (15%, 35%, 50%, 70%, 90%)
  
- **`atom_mip.png`**: Maximum intensity projection
  - Flattens 3D density to 2D by taking max along z-axis
  
- **`atom_fly.gif`**: Animated fly-through
  - Slices through the atom from bottom to top
  - ~36 frames showing 3D structure
  
- **`convergence.png`**: Convergence plot
  - Log-scale plot of relative change per iteration
  - Shows when the simulation stabilized

- **`summary.json`**: Complete experiment summary
  - Paths to all artifacts
  - Excerpt of atom constant
  - Experiment metadata

## The Atom Constant

The `atom_constant.json` file is the **key reproducibility artifact**. It contains everything needed to reconstruct the atom:

```json
{
  "name": "QCR-ATOM-V1",
  "grid": {
    "R": 1.0,
    "N": 72
  },
  "isosurface": {
    "iso": 0.35,
    "file": "atom_isomask.npy"
  },
  "qubits": {
    "z_expectations": [-0.05, -0.79, -0.42, ...],
    "n_qubits": 8
  },
  "convergence": {
    "n_steps": 120,
    "final_delta": 1e-5,
    "history": [1.0, 0.18, 0.03, ...]
  },
  "notes": "Coordinate-level atom reconstruction from synthetic quantum run."
}
```

### Why This Matters

If this conversation gets deleted, you can give just the `atom_constant.json` (and optionally the `.npy` files) to any other AI and say:

> "Reconstruct this atom"

And it will have everything it needs:
- Grid dimensions and spatial scale
- Qubit weights that define the orbital structure
- Convergence behavior showing quality
- Link to the isosurface mask for geometry

## Algorithm Details

### Density Construction

The atom density is built from:

1. **Nucleus**: Central Gaussian (fixed)
   ```
   ρ_nucleus = 0.35 × exp(-r²/(2×0.16²))
   ```

2. **Electron lobes**: One per qubit (position depends on qubit index)
   ```
   ρ_electron[k] = w[k] × exp(-|r - r_k|²/(2×0.22²))
   ```
   where `w[k] = (1 - z[k])/2` converts z-expectation to probability

3. **Total density**: Normalized sum
   ```
   ρ_total = normalize(ρ_nucleus + Σ ρ_electron[k])
   ```

### Convergence Process

The QCR iteration simulates repeated quantum measurements:

```python
for t in range(iters):
    # "Measure" the ideal density from current qubit weights
    ideal_density = density_from_qubits(z_expectations, X, Y, Z)
    
    # Blend with previous (like wavefunction collapse)
    density = (1 - smooth) × density + smooth × ideal_density
    
    # Check convergence
    change = ||density - prev_density|| / ||prev_density||
    if change < 1e-4:
        break  # converged
```

This mimics how repeated quantum measurements gradually "pin down" the system state.

## Integration with Existing Experiments

This module is designed to work with:

- **`exotic_atom_floquet.py`**: Can replace synthetic qubit weights with real Floquet results
- **`quantum_tunneling.py`**: Time-dependent densities could use QCR for each timestep
- **`quion_vizrun.py`**: QCR densities are visualization-ready

To integrate real quantum data:

```python
from quantacap.experiments.exotic_atom_floquet import run_exotic_atom
from quantacap.experiments.qcr_atom_reconstruct import run_qcr_atom

# Run Floquet calculation
floquet_result = run_exotic_atom(...)

# Extract qubit expectations
z_expectations = floquet_result['qubit_expectations']

# Reconstruct 3D atom
summary = run_qcr_atom(
    n_qubits=len(z_expectations),
    # Can pass z_expectations directly when implemented
)
```

## Future Enhancements

1. **Real quantum data input**: Load z-expectations from artifacts
2. **Marching cubes**: Generate actual 3D mesh from isosurface
3. **Time evolution**: Animate atom as qubits evolve
4. **Multi-atom systems**: Extend to molecules or lattices
5. **Export formats**: Add .obj, .stl for 3D printing

## Examples

### Quick Test (8×8×8, 3 iterations)

```bash
python -m quantacap.experiments.qcr_atom_reconstruct \
  --N 8 --iters 3
```

Output: `artifacts/qcr/` with all visualizations (completes in ~1 second)

### High Resolution (128×128×128, 200 iterations)

```bash
python -m quantacap.experiments.qcr_atom_reconstruct \
  --N 128 --iters 200 --iso 0.3
```

Output: High-detail atom with smooth isosurface (takes ~1-2 minutes)

### Reproducible Run

```bash
python -m quantacap.experiments.qcr_atom_reconstruct \
  --seed 12345
```

Re-running with the same seed produces identical results.

## Troubleshooting

### Missing visualization artifacts

If `atom_fly.gif` or `.png` files are missing:
- Check if matplotlib/PIL are installed
- Module detects these deps and skips visualization if unavailable
- Data files (`.npy`, `.json`) are always generated

### Convergence too fast

If convergence happens in 1-2 iterations:
- This is expected with synthetic data (it's self-consistent)
- Real quantum data will show more interesting dynamics
- Increase `--iters` to see full evolution

### Large file sizes

- 64³ grid: ~2-3 MB for density + isomask
- 128³ grid: ~15-20 MB for density + isomask
- Use smaller `--N` if disk space is limited

## See Also

- `exotic_atom_floquet.py`: Source of real qubit data
- `quantum_tunneling.py`: Time-dependent quantum simulations
- `quion_vizrun.py`: Advanced 3D visualization tools
