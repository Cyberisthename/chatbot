# Atom3D Quantum Simulation Suite

This repo contains a modular suite of quantum solvers for first-principles atom modeling.

The new `run_atom3d_suite.py` orchestrates the entire pipeline:

1. Hydrogen ground state at multiple resolutions (64, 128, 256, 512)
2. Hydrogen excited states (2s and 2p)
3. Helium mean-field Hartree model
4. Field perturbations (Stark and Zeeman)
5. Tomography reconstruction
6. Dashboard summary with 4K render and energy plots

Artifacts are saved under `artifacts/atom3d/` with metadata files:
- `atom_descriptor.json`
- `energy_convergence.json`
- `field_shift.json`
- `recon_metrics.json`

Each stage produces density, wavefunction, potential arrays, and glTF + MP4 visualizations.

## Running the suite

```
python run_atom3d_suite.py --full
```

You can also specify particular stages:

```
python run_atom3d_suite.py --stages ground,excited,field
```

To select custom resolutions for the ground state run:

```
python run_atom3d_suite.py --full --resolutions 128,256
```

The pipeline enforces `facts_only` physics, uses FFT-based Laplacian, and seeds all randomness with `424242`.

