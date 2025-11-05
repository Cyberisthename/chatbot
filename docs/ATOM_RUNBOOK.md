# ATOM 3D Suite Runbook

The AtomSim suite adds FFT-based split-operator solvers, visualization, and synthetic tomography for hydrogen and helium atoms. All commands are deterministic with seed `424242` and write new artifacts without overwriting existing outputs.

## Key commands

```bash
python -m atomsim.cli hyd-ground --N 256 --L 12 --steps 1200 --dt 0.002 --out artifacts/atom3d/h1s
python -m atomsim.cli hyd-excited --nlm 2,1,0 --out artifacts/atom3d/h2p
python -m atomsim.cli he-ground --steps 3000 --out artifacts/atom3d/he
python -m atomsim.cli hyd-field --mode stark --Ez 0.02 --in artifacts/atom3d/h2p --out artifacts/atom3d/h2p_E
python -m atomsim.cli hyd-tomo --in artifacts/atom3d/h1s --angles 120 --noise 0.01 --out artifacts/atom3d/h1s_tomo
```

## Artifacts produced

Each solver creates a fresh directory and never overwrites existing outputs. Expect the following files per run:

- `psi.npy` — complex wavefunction on the 3D grid (complex64)
- `density.npy` — probability density (float32)
- `potential.npy` — potential energy grid
- `energy.json` — energy history with final energy in hartree
- `mip_xy.png`, `mip_xz.png`, `mip_yz.png` — maximum-intensity projections
- `radial_compare.png` — numerical vs analytic 1s radial density (ground state)
- `mesh_isosurface.glb` — glTF isosurface at ρ = 0.2 (requires `scikit-image`)
- `orbit_spin.mp4` — 12 s camera orbit (skipped if ffmpeg unavailable)

Additional outputs:

- Helium: `psi1.npy`, `psi2.npy`, `hartree.npy`, `total_energy.json`, `ee_energy.json`
- Field perturbations: `shifts.json`, `density_comparison.png`
- Fine structure: `shifts.json` with Darwin and spin-orbit corrections
- Tomography: `sinogram.npy`, `sinogram.png`, `reconstruction.npy`, `recon_slice.png`, `metrics.json`

## Interpretation tips

- Convergence: energies in `energy.json` should monotonically decrease for imaginary-time propagation.
- Norm: wavefunctions are normalized to unity each iteration; verify with the tests or by inspection.
- Radial fidelity: `radial_compare.png` shows overlap between numeric density and analytic 1s; deviations indicate grid or step issues.
- Field shifts: `shifts.json` stores both numeric and analytic first-order estimates for Stark and Zeeman modes.
- Tomography metrics: `SSIM > 0.85`, high PSNR, and small L2 indicate accurate reconstructions (noise-free case).
