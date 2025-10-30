#!/usr/bin/env python3
"""Quick test script for quantum tunneling experiment."""

import sys
import os

# Add quantacap to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'quantacap', 'src'))

from quantacap.experiments.quantum_tunneling import simulate_tunneling, output_artifacts

def main():
    print("Running quantum tunneling simulation...")
    print("=" * 60)
    
    # Test 1: High barrier (tunneling regime)
    print("\nTest 1: Tunneling (E=2.0, V=5.0)")
    result1 = simulate_tunneling(
        n=1024,
        barrier_center=512,
        barrier_width=128,
        barrier_height=5.0,
        energy=2.0,
        steps=2000,
        dt=0.002,
    )
    print(f"  Final transmission: {result1.final_transmission:.6f}")
    print(f"  Final reflection:   {result1.final_reflection:.6f}")
    print(f"  Total probability:  {result1.total_probability[-1]:.6f}")
    
    # Test 2: Low barrier (classical regime)
    print("\nTest 2: Classical transmission (E=5.0, V=2.0)")
    result2 = simulate_tunneling(
        n=1024,
        barrier_center=512,
        barrier_width=128,
        barrier_height=2.0,
        energy=5.0,
        steps=2000,
        dt=0.002,
    )
    print(f"  Final transmission: {result2.final_transmission:.6f}")
    print(f"  Final reflection:   {result2.final_reflection:.6f}")
    print(f"  Total probability:  {result2.total_probability[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Quantum tunneling demonstration complete!")
    print("\nKey observation:")
    print(f"  - Tunneling case (E<V): {result1.final_transmission*100:.3f}% transmission")
    print(f"  - Classical case (E>V): {result2.final_transmission*100:.3f}% transmission")
    print("\nThis demonstrates quantum tunneling - particles can pass through")
    print("barriers even when classically forbidden!")

if __name__ == "__main__":
    main()
