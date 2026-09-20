import unittest
import numpy as np
import os
import sys

# Add root to sys.path for test portability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum_llm.eeg_to_tonal_engine import TonalSoulEngine, ResonanceMonitor

class TestEEGToTonal(unittest.TestCase):
    def setUp(self):
        self.engine = TonalSoulEngine()
        self.monitor = ResonanceMonitor()
        self.fs = 250
        self.t = np.linspace(0, 3, self.fs * 3)

    def test_bits_extraction(self):
        # Bits are still extracted from signal metrics (unchanged behaviour)
        ch1 = np.sin(2 * np.pi * 40 * self.t) + 0.1 * np.random.randn(self.fs * 3)
        bits = self.engine.extract_bits([ch1, ch1])
        self.assertIn("x_bits", bits)
        self.assertIn("y_bits", bits)
        self.assertIn("z_bits", bits)

    def test_bits_only_returns_no_signal(self):
        # FIXED: the old heuristic fired on bit counts alone (no spectral
        # evidence) — pure noise could inflate counts and trigger. Now a
        # bits-only call cannot fire: raw signal is required for verification.
        ch1 = np.sin(2 * np.pi * 40 * self.t) + 0.1 * np.random.randn(self.fs * 3)
        bits = self.engine.extract_bits([ch1, ch1])
        resonance = self.monitor.analyze_resonance(bits)
        self.assertFalse(resonance["resonance_detected"])
        self.assertEqual(resonance["state"], "NO_SIGNAL")

if __name__ == "__main__":
    unittest.main()