# Physics-First Atom Solver

## Overview

This module provides a **physics-first** approach to atomic structure visualization. Instead of rendering hand-tuned orbital shapes, it solves the actual Schrödinger equation from first principles using imaginary-time evolution.

## Philosophy

The key difference between this and traditional visualization approaches:

- **Traditional**: Draw what textbooks say atoms "should" look like
- **Physics-first**: Solve the wave equation and let the shape emerge naturally

This means:
1. Start from physical constants
2. Write down the Coulomb potential `V(r) = -Z/r`
3. Solve the time-independent Schrödinger equation on a 3D grid
4. Output the real electron density `|ψ|²`

## The Method: Imaginary-Time Evolution

The solver uses imaginary-time propagation to find the ground state:

```
∂ψ/∂τ = (1/2) ∇²ψ - V(r) ψ
```

In atomic units (ℏ = 1, m_e = 1, e = 1, a₀ = 1), this automatically:
- Converges to the lowest energy eigenstate
- Produces the correct quantum probability distribution
- Respects the Coulomb attraction to the nucleus

## Usage

### Basic Usage

```bash
python quantacap/src/quantacap/experiments/solve_atom_from_constants.py
```

### With Custom Parameters

```bash
python quantacap/src/quantacap/experiments/solve_atom_from_constants.py \
  --N 64 \
  --L 12.0 \
  --Z 1.0 \
  --steps 600 \
  --dt 0.002 \
  --softening 0.3
```

### Parameters

- `--N`: Grid points per axis (default: 64)
  - Higher = more resolution, more computation time
  - Use 32 for quick tests, 64+ for production

- `--L`: Physical box size in atomic units (default: 12.0)
  - Should be large enough that the wavefunction decays to near zero at edges
  - For hydrogen-like atoms, L=12 is usually sufficient

- `--Z`: Nuclear charge (default: 1.0)
  - Z=1: hydrogen
  - Z=2: helium (single electron)
  - Higher Z → more tightly bound electron

- `--steps`: Imaginary time steps (default: 600)
  - More steps → better convergence
  - Check `energy_convergence.png` to see if you need more

- `--dt`: Imaginary time step size (default: 0.002)
  - Smaller = more stable but slower
  - If energy diverges, reduce dt

- `--softening`: Nuclear softening parameter (default: 0.3)
  - Prevents singularity at r=0
  - Physically represents finite nuclear size

## Output

All outputs are saved to `artifacts/real_atom/`:

### Data Files
- `psi.npy`: 3D wavefunction array
- `density.npy`: 3D electron density (`|ψ|²`)
- `V.npy`: 3D potential field

### Visualizations
- `atom_mip.png`: Maximum intensity projection (overall shape)
- `slice_*.png`: Cross-sections through the atom
- `energy_convergence.png`: Energy vs iteration (proves convergence)

### Descriptor
- `atom_descriptor.json`: Complete specification for reproducibility
  - Grid parameters
  - Physical constants
  - Convergence history
  - Artifact paths

## The Atom Descriptor

The descriptor JSON allows any AI or tool to rebuild the exact same atom:

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

## Testing Different Atoms

### Hydrogen (default)
```bash
python solve_atom_from_constants.py --Z 1.0
```

### Helium+ (single electron)
```bash
python solve_atom_from_constants.py --Z 2.0 --L 8.0
```
(Smaller L because electron is more tightly bound)

### Lithium²⁺ (single electron)
```bash
python solve_atom_from_constants.py --Z 3.0 --L 6.0
```

### Exotic Atom (modified potential)
To test "what if atoms were slightly different", modify the `coulomb_potential` function in the code.

## Comparison with Other Modules

This module complements existing atom visualization code:

- `atom1d.py`: Simple Gaussian states for 1D/2D experiments
- `qcr_atom_reconstruct.py`: Reconstructs atoms from qubit measurements
- `solve_atom_from_constants.py`: **This module** - solves from first principles

Use this module when you need:
- The "real" quantum solution
- Validation data for other approaches
- A baseline to compare against
- Educational demonstrations of quantum mechanics

## Implementation Details

### Grid Method
Uses finite-difference discretization on a uniform Cartesian grid.

### Boundary Conditions
Implicit zero-flux boundaries (wavefunction naturally decays at edges).

### Normalization
Enforced at every step: `∫|ψ|² dV = 1`

### Energy Calculation
```
E = ⟨ψ| Ĥ |ψ⟩ = ⟨ψ| -½∇² + V |ψ⟩
```

### Convergence
Considered converged when energy changes by less than ~0.001% per step.

## Known Limitations

1. **Single Electron Only**: This is a one-body solver. Multi-electron atoms need Hartree-Fock or DFT.

2. **Ground State Only**: Finds the lowest energy state. Excited states require orthogonalization.

3. **No Spin**: Electron spin is not explicitly included (though it wouldn't affect the spatial distribution much for single electrons).

4. **Cartesian Grid**: More efficient methods exist (spherical coordinates, basis sets), but grid methods are more transparent.

## Future Extensions

Possible enhancements:
- Excited states via orthogonalization
- Multi-electron atoms via mean-field approximation
- Time-dependent dynamics
- Different potentials (harmonic oscillator, modified Coulomb, etc.)
- Export to other formats (VTK for 3D rendering, etc.)

## References

- Imaginary-time evolution: standard technique in quantum many-body physics
- Atomic units: simplify equations by setting fundamental constants to 1
- Grid methods: Chapter 9 of "Computational Physics" by Newman

## Example Output

Expected ground state energy for hydrogen (atomic units):
- Exact: E = -0.5
- This solver (N=64, L=12): E ≈ -0.48 to -0.49

(Small error due to finite grid spacing and softening)
