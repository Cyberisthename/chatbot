# 3D Atom Solver Artifacts

This directory contains the output from the 3D Schrödinger atom solver.

## Included Files

✅ **64³ resolution** (2.1 MB each):
- `density_N64.npy` - Electron density
- `psi_N64.npy` - Wavefunction
- `potential_N64.npy` - Coulomb potential

✅ **128³ resolution** (17 MB each):
- `density_N128.npy` - Electron density
- `psi_N128.npy` - Wavefunction
- `potential_N128.npy` - Coulomb potential

✅ **Renders** (4K-ready):
- `atom_mip_xy.png` - Top view (max projection along Z)
- `atom_mip_xz.png` - Side view (max projection along Y)
- `atom_mip_yz.png` - Other side (max projection along X)

✅ **Energy convergence**:
- `energy_N64.json` - 64³ stage
- `energy_N128.json` - 128³ stage
- `energy_N256.json` - 256³ stage

✅ **Metadata**:
- `atom3d_descriptor.json` - Complete configuration and results

## Missing Files (Too Large for Git)

⚠️ **256³ resolution files** (128 MB each) are NOT included in the repository:
- `density_N256.npy`
- `psi_N256.npy`
- `potential_N256.npy`

These files exceed GitHub's 100MB file size limit.

## Regenerating 256³ Files

To regenerate the complete dataset including 256³ resolution:

```bash
# Remove existing artifacts
rm -rf artifacts/real_atom_3d/

# Run the full solver
python -m experiments.solve_atom_3d_discovery
```

This will take ~20 minutes and generate all files including the 256³ resolution data (~440 MB total).

## Viewing Results

```bash
# List all files
ls -lh artifacts/real_atom_3d/

# View metadata
cat artifacts/real_atom_3d/atom3d_descriptor.json

# View images (if you have an image viewer)
# artifacts/real_atom_3d/atom_mip_xy.png
# artifacts/real_atom_3d/atom_mip_xz.png
# artifacts/real_atom_3d/atom_mip_yz.png
```

## Loading Data in Python

```python
import numpy as np
import json

# Load wavefunction
psi_128 = np.load("artifacts/real_atom_3d/psi_N128.npy")
print(f"Wavefunction shape: {psi_128.shape}")

# Load density
density_128 = np.load("artifacts/real_atom_3d/density_N128.npy")
print(f"Density shape: {density_128.shape}")

# Load metadata
with open("artifacts/real_atom_3d/atom3d_descriptor.json") as f:
    metadata = json.load(f)
    print(f"Final energy (128³): {metadata['stages'][1]['final_energy']} a.u.")
```

## File Sizes

| Resolution | Files per stage | Size per file | Total |
|-----------|----------------|---------------|--------|
| 64³ | 3 (psi, density, potential) | 2.1 MB | ~6 MB |
| 128³ | 3 (psi, density, potential) | 17 MB | ~51 MB |
| 256³ | 3 (psi, density, potential) | 128 MB | ~384 MB |
| **Total (with 256³)** | | | **~440 MB** |
| **Total (without 256³)** | | | **~58 MB** |

The renders and JSON files are small (< 2 MB total).

## Notes

- All data is in atomic units (ħ = m_e = e = 1)
- Grid spacing: dx = 12.0/N atomic units
- Physical domain: [-6, 6]³ atomic units
- Hydrogen atom (Z=1) ground state

For more details, see:
- `experiments/ATOM_3D_DISCOVERY_README.md`
- `ATOM_3D_DISCOVERY_SUMMARY.md`
