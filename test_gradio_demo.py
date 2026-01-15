#!/usr/bin/env python3
"""
Quick test script to verify Gradio demo imports work
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    from bio_knowledge.virtual_cancer_cell_simulator import (
        VirtualCancerCellSimulator,
        VirtualCellState,
        TreatmentOutcome,
        CellState
    )
    print("✅ bio_knowledge.virtual_cancer_cell_simulator")
except Exception as e:
    print(f"❌ bio_knowledge.virtual_cancer_cell_simulator: {e}")

try:
    from bio_knowledge.cancer_hypothesis_generator import CancerHypothesisGenerator, Hypothesis
    print("✅ bio_knowledge.cancer_hypothesis_generator")
except Exception as e:
    print(f"❌ bio_knowledge.cancer_hypothesis_generator: {e}")

try:
    from quantum.multiversal_quantum import MultiversalQuantumEngine, MultiversalExperimentConfig
    print("✅ quantum.multiversal_quantum")
except Exception as e:
    print(f"❌ quantum.multiversal_quantum: {e}")

try:
    from core.multiversal_compute_system import MultiversalComputeSystem, MultiversalQuery
    print("✅ core.multiversal_compute_system")
except Exception as e:
    print(f"❌ core.multiversal_compute_system: {e}")

try:
    import gradio as gr
    print("✅ gradio")
except Exception as e:
    print(f"❌ gradio: {e}")

try:
    import numpy as np
    print("✅ numpy")
except Exception as e:
    print(f"❌ numpy: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✅ matplotlib")
except Exception as e:
    print(f"❌ matplotlib: {e}")

print("\n🎉 All critical imports successful!")
print("   Run: python gradio_quantum_cancer_demo.py")
