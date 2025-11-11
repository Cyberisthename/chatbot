#!/usr/bin/env python3
"""
Validation Script for 3D Atom Discovery Results
================================================

This script validates the numerical results from the 3D Schrödinger solver:
1. Checks that all required files exist
2. Verifies energy convergence
3. Validates normalization
4. Checks for spherical symmetry
5. Compares to theoretical expectations

Run: python validate_atom_3d_results.py
"""

import json
import os
from pathlib import Path

import numpy as np


def check_files_exist():
    """Check that all expected artifact files exist."""
    print("="*60)
    print("FILE EXISTENCE CHECK")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    
    required_files = [
        "atom3d_descriptor.json",
        "atom_mip_xy.png",
        "atom_mip_xz.png", 
        "atom_mip_yz.png",
        "atom_spin.gif",
        "density_N64.npy",
        "density_N128.npy",
        "density_N256.npy",
        "psi_N64.npy",
        "psi_N128.npy",
        "psi_N256.npy",
        "potential_N64.npy",
        "potential_N128.npy",
        "potential_N256.npy",
        "energy_N64.json",
        "energy_N128.json",
        "energy_N256.json",
    ]
    
    all_exist = True
    for fname in required_files:
        fpath = base_dir / fname
        exists = fpath.exists()
        status = "✅" if exists else "❌"
        size = f"({fpath.stat().st_size / 1024**2:.1f} MB)" if exists else ""
        print(f"{status} {fname} {size}")
        if not exists:
            all_exist = False
    
    print()
    if all_exist:
        print("✅ All required files present!")
    else:
        print("❌ Some files are missing!")
    
    return all_exist


def validate_energy_convergence():
    """Validate that energy converges monotonically downward."""
    print("\n" + "="*60)
    print("ENERGY CONVERGENCE VALIDATION")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    
    theoretical_ground_state = -0.5  # hartree for hydrogen
    
    for stage in [64, 128, 256]:
        energy_file = base_dir / f"energy_N{stage}.json"
        
        with open(energy_file) as f:
            energy_data = json.load(f)
        
        energies = [e["E"] for e in energy_data]
        steps = [e["step"] for e in energy_data]
        
        # Check monotonic decrease
        is_decreasing = all(energies[i] <= energies[i-1] for i in range(1, len(energies)))
        
        # Calculate convergence
        final_energy = energies[-1]
        accuracy = abs(final_energy / theoretical_ground_state) * 100
        
        status = "✅" if is_decreasing else "⚠️"
        
        print(f"\n{status} Stage N={stage}³:")
        print(f"   Initial energy:  {energies[0]:+.6f} hartree")
        print(f"   Final energy:    {energies[-1]:+.6f} hartree")
        print(f"   Energy decrease: {abs(energies[-1] - energies[0]):.6f} hartree")
        print(f"   Monotonic decay: {is_decreasing}")
        print(f"   Accuracy vs H:   {accuracy:.1f}%")
        
        if not is_decreasing:
            print(f"   ⚠️  WARNING: Energy not monotonically decreasing!")


def validate_normalization():
    """Check that wavefunctions are properly normalized."""
    print("\n" + "="*60)
    print("NORMALIZATION VALIDATION")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    box_size = 12.0
    
    for stage, N in [(64, 64), (128, 128), (256, 256)]:
        psi_file = base_dir / f"psi_N{stage}.npy"
        density_file = base_dir / f"density_N{stage}.npy"
        
        psi = np.load(psi_file)
        density = np.load(density_file)
        
        dx = box_size / N
        dV = dx**3
        
        # Check |ψ|² = ρ
        computed_density = np.abs(psi)**2
        density_match = np.allclose(computed_density, density, rtol=1e-5)
        
        # Check normalization ∫|ψ|² dV = 1
        norm_psi = (np.abs(psi)**2).sum() * dV
        norm_density = density.sum() * dV
        
        # Tolerance for normalization (should be very close to 1.0)
        is_normalized = abs(norm_density - 1.0) < 0.01
        
        status = "✅" if (density_match and is_normalized) else "⚠️"
        
        print(f"\n{status} Stage N={stage}³:")
        print(f"   ρ = |ψ|² match:   {density_match}")
        print(f"   ∫|ψ|² dV =        {norm_psi:.6f}")
        print(f"   ∫ρ dV =           {norm_density:.6f}")
        print(f"   Normalized:       {is_normalized}")
        
        if not is_normalized:
            print(f"   ⚠️  WARNING: Normalization error = {abs(norm_density - 1.0):.4f}")


def validate_symmetry():
    """Check for spherical symmetry in the density."""
    print("\n" + "="*60)
    print("SYMMETRY VALIDATION")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    
    # Only check highest resolution
    density = np.load(base_dir / "density_N256.npy")
    N = 256
    
    # Check symmetry by comparing octants
    center = N // 2
    
    # Extract 8 octants
    octants = [
        density[center:, center:, center:],     # (+,+,+)
        density[center:, center:, :center],     # (+,+,-)
        density[center:, :center, center:],     # (+,-,+)
        density[center:, :center, :center],     # (+,-,-)
        density[:center, center:, center:],     # (-,+,+)
        density[:center, center:, :center],     # (-,+,-)
        density[:center, :center, center:],     # (-,-,+)
        density[:center, :center, :center],     # (-,-,-)
    ]
    
    # Flip octants to align them all
    flipped = [
        octants[0],
        np.flip(octants[1], axis=2),
        np.flip(octants[2], axis=1),
        np.flip(np.flip(octants[3], axis=1), axis=2),
        np.flip(octants[4], axis=0),
        np.flip(np.flip(octants[5], axis=0), axis=2),
        np.flip(np.flip(octants[6], axis=0), axis=1),
        np.flip(np.flip(np.flip(octants[7], axis=0), axis=1), axis=2),
    ]
    
    # Compare all octants to first one
    reference = flipped[0]
    symmetry_scores = []
    
    for i, octant in enumerate(flipped[1:], 1):
        # Compute relative difference
        diff = np.abs(octant - reference)
        rel_diff = (diff / (reference.max() + 1e-10)).mean()
        symmetry_scores.append(rel_diff)
    
    avg_asymmetry = np.mean(symmetry_scores)
    max_asymmetry = np.max(symmetry_scores)
    
    is_symmetric = avg_asymmetry < 0.1  # 10% threshold
    
    status = "✅" if is_symmetric else "⚠️"
    
    print(f"\n{status} Spherical Symmetry (256³ resolution):")
    print(f"   Average asymmetry:  {avg_asymmetry*100:.2f}%")
    print(f"   Maximum asymmetry:  {max_asymmetry*100:.2f}%")
    print(f"   Is spherical:       {is_symmetric}")
    
    if is_symmetric:
        print(f"   ✅ Density is spherically symmetric (ground state)")
    else:
        print(f"   ⚠️  Non-spherical structure detected!")
        print(f"   → Could indicate excited state or numerical artifact")
    
    return is_symmetric


def analyze_radial_distribution():
    """Analyze radial density distribution."""
    print("\n" + "="*60)
    print("RADIAL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    density = np.load(base_dir / "density_N256.npy")
    
    N = 256
    L = 12.0
    x = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Find peak density radius
    max_idx = np.unravel_index(density.argmax(), density.shape)
    max_r = R[max_idx]
    max_density = density.max()
    
    # Find effective radius (where density drops to 1/e of max)
    threshold = max_density / np.e
    effective_radius = R[density > threshold].max()
    
    # Bohr radius for comparison
    bohr_radius = 1.0  # atomic units
    
    print(f"\n   Peak density:          {max_density:.6f}")
    print(f"   Peak radius:           {max_r:.3f} a.u.")
    print(f"   Effective radius:      {effective_radius:.3f} a.u.")
    print(f"   Expected (Bohr):       {bohr_radius:.3f} a.u.")
    print(f"   Ratio to Bohr:         {effective_radius/bohr_radius:.2f}×")
    
    # Total probability within different radii
    for r_cutoff in [1.0, 2.0, 3.0, 5.0]:
        prob = density[R < r_cutoff].sum() * (L/N)**3
        print(f"   Probability (r<{r_cutoff}):   {prob:.1%}")


def print_summary():
    """Print overall summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    base_dir = Path("artifacts/real_atom_3d")
    
    # Load descriptor
    with open(base_dir / "atom3d_descriptor.json") as f:
        descriptor = json.load(f)
    
    print(f"\nExperiment: {descriptor['name']}")
    print(f"Mode: FACTS_ONLY = {descriptor['facts_only']}")
    print(f"\nConfiguration:")
    print(f"  Nuclear charge Z = {descriptor['config']['Z']}")
    print(f"  Box size = {descriptor['config']['box']} a.u.")
    print(f"  Softening = {descriptor['config']['softening']}")
    print(f"  Resolutions = {descriptor['config']['N_stages']}")
    print(f"  Steps per stage = {descriptor['config']['steps_per_stage']}")
    
    print(f"\nFinal Results:")
    for stage_data in descriptor['stages']:
        N = stage_data['N']
        E = stage_data['final_energy']
        accuracy = abs(E / -0.5) * 100
        print(f"  {N}³: E = {E:.6f} hartree ({accuracy:.1f}% of theoretical)")
    
    print(f"\n✅ All validations complete!")
    print(f"\nView results:")
    print(f"  artifacts/real_atom_3d/atom_mip_xy.png")
    print(f"  artifacts/real_atom_3d/atom_mip_xz.png")
    print(f"  artifacts/real_atom_3d/atom_mip_yz.png")
    print(f"  artifacts/real_atom_3d/atom_spin.gif")
    print("="*60 + "\n")


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("3D ATOM DISCOVERY - RESULTS VALIDATION")
    print("="*60 + "\n")
    
    # Run all checks
    files_ok = check_files_exist()
    
    if not files_ok:
        print("\n❌ Cannot proceed - missing required files!")
        return
    
    validate_energy_convergence()
    validate_normalization()
    validate_symmetry()
    analyze_radial_distribution()
    print_summary()


if __name__ == "__main__":
    main()
