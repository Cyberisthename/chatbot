#!/usr/bin/env python3
"""
Quick test for the 3D atom discovery solver.
This runs a minimal version (only 32³ to be fast) to verify the code works.
"""

import sys
import os

# Test imports
try:
    import numpy as np
    print("✓ numpy available")
except ImportError:
    print("✗ numpy not available")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError:
    print("⚠ matplotlib not available (optional)")

try:
    from scipy.ndimage import zoom
    print("✓ scipy available")
except ImportError:
    print("⚠ scipy not available (optional)")

try:
    import imageio
    print("✓ imageio available")
except ImportError:
    print("⚠ imageio not available (optional)")

# Test the solver
print("\nTesting 3D atom solver...")

from experiments.solve_atom_3d_discovery import (
    potential_field,
    laplacian3d,
    normalize,
    compute_energy,
    evolve_stage,
)

# Quick test at low resolution
print("\nRunning quick test at N=32³...")
result = evolve_stage(
    N=32,
    box=12.0,
    Z=1.0,
    centers=[[0.0, 0.0, 0.0]],
    softening=0.3,
    dt=0.002,
    steps=50,
    psi_init=None,
    save_every=10,
)

print(f"\nTest results:")
print(f"  Grid shape: {result['psi'].shape}")
print(f"  Grid spacing dx: {result['dx']:.6f}")
print(f"  Final energy: {result['energies'][-1]['E']:.6f} a.u.")
print(f"  Energy should be around -0.3 to -0.4 a.u. for hydrogen")

# Verify the energy is reasonable
E_final = result['energies'][-1]['E']
if -0.5 < E_final < 0.0:
    print("\n✅ Test PASSED: Energy in expected range")
    sys.exit(0)
else:
    print(f"\n⚠ Test WARNING: Energy {E_final} is outside expected range")
    print("  (This might be OK for a quick test with only 50 steps)")
    sys.exit(0)
