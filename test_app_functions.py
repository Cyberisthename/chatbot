#!/usr/bin/env python3
"""
Test the core functions in app.py without running Streamlit.
"""

import sys
import numpy as np

# Mock streamlit for testing
class MockStreamlit:
    def set_page_config(self, **kwargs): pass
    def title(self, text): pass
    def caption(self, text): pass
    def divider(self): pass
    def tabs(self, names): return [None] * len(names)
    def subheader(self, text): pass
    def markdown(self, text): pass
    def slider(self, *args, **kwargs): return args[3] if len(args) > 3 else 0
    def button(self, text): return False
    def spinner(self, text): return self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def success(self, text): pass
    def warning(self, text): pass
    def error(self, text): pass
    def write(self, obj): pass
    def metric(self, label, value): pass
    def pyplot(self, fig, **kwargs): pass
    def download_button(self, *args, **kwargs): pass

sys.modules['streamlit'] = MockStreamlit()

# Import functions from app
print("Importing app.py functions...")
from app import (
    solve_atom_3d,
    double_slit_sim,
    field_interference_sim,
    chsh_sim,
    relativistic_graph,
    holo_entropy,
    new_rng,
    SEED,
)

print(f"✓ App functions imported successfully (seed={SEED})")

def test_atom_3d():
    print("\n=== Test 1: Atom 3D Solver ===")
    result = solve_atom_3d(N=32, box=12.0, steps=50, dt=0.002)
    print(f"  Final energy: {result['final_energy']:.6f} Hartree")
    print(f"  Density shape: {result['density'].shape}")
    print(f"  Energy convergence: {len(result['energies'])} steps")
    assert result['density'].shape == (32, 32, 32)
    assert len(result['energies']) > 0
    print("  ✓ Passed")

def test_double_slit():
    print("\n=== Test 2: Double-Slit ===")
    result = double_slit_sim(N=256, k=20.0)
    print(f"  Visibility: {result['visibility']:.4f}")
    print(f"  Intensity shape: {result['intensity_interference'].shape}")
    assert result['visibility'] > 0
    assert len(result['x']) == 256
    print("  ✓ Passed")

def test_field_interference():
    print("\n=== Test 3: Field Interference ===")
    result = field_interference_sim(N=128, T=100, src=2)
    print(f"  Mean visibility: {result['visibility']:.4f}")
    print(f"  Field shape: {result['phi'].shape}")
    print(f"  Sources: {len(result['source_locs'])}")
    assert result['phi'].shape == (128, 128)
    assert len(result['source_locs']) == 2
    print("  ✓ Passed")

def test_chsh():
    print("\n=== Test 4: CHSH Bell Inequality ===")
    E, S = chsh_sim(shots=10000, depol=0.0)
    print(f"  S parameter: {S:.4f}")
    print(f"  Correlators: {[f'{e:.4f}' for e in E]}")
    assert len(E) == 4
    assert S > 2.0, f"Expected S > 2 for Bell violation, got {S:.4f}"
    print("  ✓ Passed (quantum violation)")

def test_relativistic_graph():
    print("\n=== Test 5: Relativistic Graph ===")
    dag, dur_n, Tn, Tp = relativistic_graph(n=16, beta=0.6)
    print(f"  Newtonian duration: {dur_n:.4f}")
    print(f"  Relativistic duration: {Tp[-1]:.4f}")
    print(f"  Edges: {len(dag)}")
    assert len(Tn) == 16
    assert len(Tp) == 16
    print("  ✓ Passed")

def test_holo_entropy():
    print("\n=== Test 6: Holographic Entropy ===")
    local_rng = new_rng()
    cube = local_rng.normal(size=(48, 48, 48))
    cube = (cube > 0).astype(np.uint8)
    areas, ent, coeff = holo_entropy(cube)
    print(f"  Samples: {len(areas)}")
    print(f"  Fit: H = {coeff[0]:.6f} * A + {coeff[1]:.6f}")
    assert len(areas) > 0
    assert len(ent) == len(areas)
    print("  ✓ Passed")

def main():
    print("\n" + "="*60)
    print("Testing Infinite Compute Lab App Functions")
    print("="*60)
    
    try:
        test_atom_3d()
        test_double_slit()
        test_field_interference()
        test_chsh()
        test_relativistic_graph()
        test_holo_entropy()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nReady to run: streamlit run app.py")
        print("="*60 + "\n")
        
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
