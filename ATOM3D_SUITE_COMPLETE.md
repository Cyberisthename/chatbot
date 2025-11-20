# Atom3D Quantum Simulation Suite - Complete Implementation

## Overview

This repository contains a comprehensive first-principles quantum atom simulation suite that uses **no pre-defined orbitals** or analytic solutions. All results are computed from the time-dependent Schrödinger equation using imaginary-time propagation with FFT-based Laplacian operators.

The suite has been enhanced with a unified orchestration script that runs the complete multi-stage pipeline:

## Features Implemented

### ✅ Core Physics
- **Imaginary-time Schrödinger equation** with split-operator FFT method
- **FFT-based Laplacian** for optimal numerical accuracy
- **Deterministic seed** (424242) for reproducibility
- **Facts-only mode**: No smoothing, no symmetrization, no precomputed solutions

### ✅ Atomic Systems
1. **Hydrogen Ground State** (1s)
   - Multiple resolutions: 64³, 128³, 256³, 512³
   - Progressive refinement with upsampling
   - Energy convergence tracking

2. **Hydrogen Excited States**
   - 2s orbital (n=2, l=0, m=0)
   - 2p orbital (n=2, l=1, m=0)
   - Orthogonal projection against ground state

3. **Helium Atom** (Hartree mean-field)
   - Two-electron self-consistent field
   - Electron-electron repulsion via Hartree potential
   - Singlet and triplet configurations

### ✅ Field Perturbations
- **Stark Effect**: Linear electric field perturbation
- **Zeeman Effect**: Magnetic field splitting
- Numerical shift calculations with comparison to analytic predictions

### ✅ Inverse Problem
- **Tomography Reconstruction**: Filtered back-projection from synthetic projections
- **Quality Metrics**: SSIM, PSNR, L2 norm

### ✅ Visualization & Analysis
- **Multi-isosurface 4K renders** (REAL-ATOM-3D-DISCOVERY-V2.png)
- **Energy convergence plots** for all stages
- **3D mesh exports** (glTF format)
- **Orbit spin animations** (MP4 format)
- **Radial density profiles** with analytic comparison
- **Maximum intensity projections** (XY, XZ, YZ views)

### ✅ Metadata & Discovery
- `atom_descriptor.json` — Master descriptor with run metadata
- `energy_convergence.json` — Complete energy histories
- `field_shift.json` — Stark and Zeeman shift data
- `recon_metrics.json` — Tomography reconstruction metrics
- **Discovery candidate detection**: Automatic flagging of unexpected asymmetries

## Usage

### Option 1: Full Automated Suite

Run everything with default settings:

```bash
python run_atom3d_suite.py --full
```

Or use the convenience script:

```bash
./run_full_suite.sh
```

### Option 2: Selective Stages

Run only specific stages:

```bash
python run_atom3d_suite.py --stages ground,excited,helium
```

### Option 3: Custom Resolutions

For faster testing with lower resolutions:

```bash
python run_atom3d_suite.py --full --resolutions 64,128,256
```

### Option 4: Manual CLI

Each simulation can also be run individually using the modular CLI:

```bash
# Hydrogen ground state
python -m atomsim.cli hyd-ground --N 256 --L 12.0 --steps 1200 --out artifacts/atom3d/hyd-ground-N256

# Hydrogen excited 2p
python -m atomsim.cli hyd-excited --N 256 --nlm 2,1,0 --steps 2000 --out artifacts/atom3d/hyd-excited-2p

# Helium ground state
python -m atomsim.cli he-ground --N 256 --steps 3000 --out artifacts/atom3d/he-ground

# Stark effect
python -m atomsim.cli hyd-field --mode stark --in artifacts/atom3d/hyd-ground-N256 --Ez 0.01 --out artifacts/atom3d/hyd-field-stark

# Tomography
python -m atomsim.cli hyd-tomo --in artifacts/atom3d/hyd-ground-N256 --angles 90 --noise 0.02 --out artifacts/atom3d/hyd-tomo
```

## Output Structure

```
artifacts/atom3d/
├── atom_descriptor.json           # Master descriptor
├── energy_convergence.json        # All energy histories
├── field_shift.json               # Field perturbation data
├── recon_metrics.json             # Tomography metrics
├── REAL-ATOM-3D-DISCOVERY-V2.png  # 4K composite render
├── dashboard/
│   ├── atom_descriptor.json
│   ├── energy_convergence.png
│   └── ...
├── hyd-ground-N64/
│   ├── density.npy
│   ├── psi.npy
│   ├── potential.npy
│   ├── energy.json
│   ├── mesh_isosurface.glb
│   ├── orbit_spin.mp4
│   ├── mip_xy.png
│   ├── mip_xz.png
│   └── mip_yz.png
├── hyd-ground-N128/
│   └── ...
├── hyd-ground-N256/
│   └── ...
├── hyd-excited-2s/
│   └── ...
├── hyd-excited-2p/
│   └── ...
├── he-ground/
│   ├── psi1.npy
│   ├── psi2.npy
│   ├── density.npy
│   ├── potential.npy
│   ├── hartree.npy
│   ├── total_energy.json
│   └── ...
├── hyd-field-stark/
│   ├── shifts.json
│   └── ...
├── hyd-field-zeeman/
│   ├── shifts.json
│   └── ...
└── hyd-tomo/
    ├── metrics.json
    ├── sinogram.npy
    ├── reconstruction.npy
    └── ...
```

## Configuration

The suite is configured via `CONFIG` dict in `run_atom3d_suite.py`:

```python
CONFIG = {
    "seed": 424242,                # Deterministic seed
    "box_size": 12.0,              # Simulation domain size (Bohr radii)
    "softening": 0.3,              # Nuclear softening parameter
    "resolutions": [64, 128, 256, 512],  # Grid resolutions
    "dt": 0.002,                   # Time step
    "convergence_tol": 1e-6,       # Energy convergence threshold
    "max_steps": 5000,             # Maximum iterations
    "output_dir": "artifacts/atom3d",
}
```

## Dependencies

Required Python packages:
- `numpy` — Array operations
- `scipy` — FFT, special functions, interpolation
- `matplotlib` — Plotting and visualization
- `imageio` — MP4 video generation
- `scikit-image` — Marching cubes isosurface extraction

Install all dependencies:

```bash
pip install numpy scipy matplotlib imageio scikit-image
```

## Technical Details

### Physics Model

**Hamiltonian** (atomic units):
```
H = -½∇² + V(r)
```

**Potential** (softened Coulomb):
```
V(r) = -Z / √(r² + ε²)
```

**Evolution operator** (Strang splitting):
```
exp(-Ĥ dt) ≈ exp(-½T̂ dt) exp(-V̂ dt) exp(-½T̂ dt)
```

**Kinetic propagator** (Fourier space):
```
exp(-½T̂ dt) ψ̂(k) = exp(-½|k|² dt) ψ̂(k)
```

### Numerical Accuracy

- **Grid spacing**: dx = L / N
- **Spectral accuracy**: FFT-based derivatives
- **Absorbing boundaries**: cos⁸ mask in outer 10%
- **Normalization**: ∫|ψ|² dV = 1 enforced at each step
- **Convergence**: Adaptive time-stepping if energy increases

### Discovery Metrics

The suite automatically analyzes each density for:
- **Anisotropy**: Deviation from spherical symmetry
- **Asymmetry**: Mirror symmetry violations
- **Nodal structure**: Regions of low density
- **Candidate flagging**: Unexpected patterns for further investigation

## References

### Theory
- R. Kosloff, "Time-dependent quantum-mechanical methods for molecular dynamics," J. Phys. Chem. 92, 2087 (1988)
- M. D. Feit, J. A. Fleck, Jr., and A. Steiger, "Solution of the Schrödinger equation by a spectral method," J. Comput. Phys. 47, 412 (1982)

### Method
- Split-operator FFT method for time evolution
- Hartree mean-field approximation for multi-electron systems
- Filtered back-projection for tomographic reconstruction

## License

See LICENSE file in the repository.

## Authors

Developed as part of the Quantacap quantum research initiative.

---

**Note**: This is a research-grade physics simulation tool. Results should be validated against experimental data or higher-order quantum chemistry methods for production applications.
