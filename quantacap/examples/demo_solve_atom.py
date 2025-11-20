#!/usr/bin/env python
"""
Demo: Physics-First Atom Solver

This demonstrates how to use the Schrödinger equation solver
to compute atomic wavefunctions from first principles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantacap.experiments.solve_atom_from_constants import imaginary_time_solve
import numpy as np


def demo_hydrogen():
    print("=" * 60)
    print("Solving hydrogen atom from Schrödinger equation")
    print("=" * 60)
    
    result = imaginary_time_solve(
        N=32,
        L=10.0,
        Z=1.0,
        dt=0.002,
        steps=200,
        softening=0.3,
    )
    
    energies = [e for (_, e) in result["energies"]]
    print(f"\nInitial energy: {energies[0]:.6f} a.u.")
    print(f"Final energy:   {energies[-1]:.6f} a.u.")
    print(f"Exact (theory): -0.500000 a.u.")
    print(f"Error:          {abs(energies[-1] + 0.5):.6f} a.u.")
    
    density = result["density"]
    psi = result["psi"]
    
    print(f"\nWavefunction shape: {psi.shape}")
    print(f"Density integral:   {np.sum(density) * (result['dx']**3):.6f}")
    print(f"Max density at center: {density[16, 16, 16]:.6f}")
    
    print("\n✅ Hydrogen atom solved successfully!")
    return result


def demo_helium_plus():
    print("\n" + "=" * 60)
    print("Solving helium+ ion (He⁺, Z=2)")
    print("=" * 60)
    
    result = imaginary_time_solve(
        N=32,
        L=8.0,
        Z=2.0,
        dt=0.001,
        steps=200,
        softening=0.3,
    )
    
    energies = [e for (_, e) in result["energies"]]
    print(f"\nInitial energy: {energies[0]:.6f} a.u.")
    print(f"Final energy:   {energies[-1]:.6f} a.u.")
    print(f"Exact (theory): -2.000000 a.u.")
    print(f"Error:          {abs(energies[-1] + 2.0):.6f} a.u.")
    
    print("\n✅ Helium+ ion solved successfully!")
    return result


def compare_electron_distributions():
    print("\n" + "=" * 60)
    print("Comparing electron distributions")
    print("=" * 60)
    
    h_result = imaginary_time_solve(N=32, L=10.0, Z=1.0, dt=0.002, steps=100, softening=0.3)
    he_result = imaginary_time_solve(N=32, L=8.0, Z=2.0, dt=0.001, steps=100, softening=0.3)
    
    h_density = h_result["density"]
    he_density = he_result["density"]
    
    center = 16
    
    h_center = h_density[center, center, center]
    he_center = he_density[center, center, center]
    
    print(f"\nHydrogen (Z=1) center density:  {h_center:.6f}")
    print(f"Helium+ (Z=2) center density:   {he_center:.6f}")
    print(f"Ratio (He+/H):                   {he_center/h_center:.2f}x")
    print("\n(He+ electron is more tightly bound → higher density at nucleus)")


if __name__ == "__main__":
    demo_hydrogen()
    demo_helium_plus()
    compare_electron_distributions()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("See artifacts/real_atom/ for detailed outputs")
    print("=" * 60)
