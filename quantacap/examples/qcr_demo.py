#!/usr/bin/env python3
"""
QCR (Quantum Coordinate Reconstruction) Atom Demo

This example demonstrates how to use the QCR atom reconstruction module
to create a 3D atomic density from synthetic qubit measurements.
"""

import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantacap.experiments.qcr_atom_reconstruct import run_qcr_atom


def main():
    print("=" * 60)
    print("QCR Atom Reconstruction Demo")
    print("=" * 60)
    print()
    
    print("Running QCR atom reconstruction with:")
    print("  - Grid: 48×48×48 voxels")
    print("  - Spatial extent: ±1.0 a.u.")
    print("  - Iterations: 50")
    print("  - Isosurface: 0.35")
    print("  - Qubits: 8 (synthetic)")
    print()
    
    # Run the reconstruction
    summary = run_qcr_atom(
        N=48,
        R=1.0,
        iters=50,
        iso=0.35,
        n_qubits=8,
        seed=424242,
    )
    
    print("✓ Reconstruction complete!")
    print()
    print("Artifacts saved to:")
    for key, path in summary["artifacts"].items():
        if path:
            print(f"  - {key}: {path}")
    
    print()
    print("Atom constant (reproducibility spec):")
    const = summary["constant_excerpt"]
    print(f"  - Name: {const['name']}")
    print(f"  - Grid: {const['grid']['N']}×{const['grid']['N']}×{const['grid']['N']}")
    print(f"  - Convergence: {const['convergence']['n_steps']} steps")
    print(f"  - Final Δ: {const['convergence']['final_delta']:.2e}")
    
    print()
    print("You can now:")
    print("  1. View the slices: artifacts/qcr/slice_*.png")
    print("  2. Watch the fly-through: artifacts/qcr/atom_fly.gif")
    print("  3. Check convergence: artifacts/qcr/convergence.png")
    print("  4. Load the density: np.load('artifacts/qcr/atom_density.npy')")
    print("  5. Share the constant: artifacts/qcr/atom_constant.json")
    

if __name__ == "__main__":
    main()
