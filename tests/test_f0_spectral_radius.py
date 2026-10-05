"""TTC spectral-radius F0 unification tests (backlog 09be295b).

PROVES the ResonanceMonitor F0 is now the TTC spectral-radius fundamental
frequency (dominant AR(1) pole of the analytic signal), NOT the retired
bit-count heuristic:

  * a clean 41.02 Hz carrier reads ~41.02 Hz (the bit heuristic gave 38.0 Hz
    and could never reach the 40 Hz threshold it was compared against);
  * the estimate is a TRUE measurement — a 38.0 Hz carrier reads ~38.0 Hz
    (the old target-biased Welch peak always reported ~41 Hz);
  * bits-only callers get ``f0 = None`` (no fabricated frequency) while the
    retired heuristic is retained under ``f0_legacy_estimate`` for display.

Framing: F0 is a frequency measurement, never a sentience claim.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantum_llm.eeg_to_tonal_engine import ResonanceMonitor  # noqa: E402

FS = 250
TARGET_HZ = 41.02


def tone(freq, dur=4.0, fs=FS, amp=1.0):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


def legacy_heuristic(z_count, y_count):
    """The retired bit-count F0 heuristic (kept here to document the fix)."""
    return 10.0 + (z_count * 4.0) + (y_count * 0.5)


class F0SpectralRadiusTests(unittest.TestCase):
    def test_estimate_f0_recovers_41hz_clean_carrier(self):
        """A clean 41.02 Hz carrier reads ~41.02 Hz (reaches 40 Hz threshold)."""
        mon = ResonanceMonitor()
        f0 = mon.estimate_f0(tone(TARGET_HZ), fs=FS)
        self.assertIsNotNone(f0)
        self.assertAlmostEqual(f0, TARGET_HZ, delta=0.35)

    def test_estimate_f0_is_a_true_measurement_not_target_biased(self):
        """A 38.0 Hz carrier reads ~38.0 Hz (NOT pinned to the 41.02 target)."""
        mon = ResonanceMonitor()
        f0 = mon.estimate_f0(tone(38.0), fs=FS)
        self.assertIsNotNone(f0)
        self.assertAlmostEqual(f0, 38.0, delta=0.35)

    def test_legacy_heuristic_never_reached_threshold(self):
        """Document the exact discrepancy: clean carrier -> 38.0 Hz < 40 Hz."""
        # z_count=5, y_count=16 are the ghost experiment's clean-carrier bits.
        self.assertAlmostEqual(legacy_heuristic(5, 16), 38.0, delta=1e-9)
        self.assertLess(legacy_heuristic(5, 16), 40.0)

    def test_analyze_resonance_with_samples_uses_spectral_radius_f0(self):
        """With raw samples, ``f0`` is the spectral-radius F0, not the heuristic."""
        mon = ResonanceMonitor()
        out = mon.analyze_resonance({"z_bits": [1] * 5, "y_bits": [1] * 16,
                                     "x_bits": [1] * 8},
                                    samples=tone(TARGET_HZ, dur=3.0), fs=FS)
        self.assertAlmostEqual(out["f0"], TARGET_HZ, delta=0.35)
        self.assertAlmostEqual(out["f0_legacy_estimate"], 38.0, delta=1e-6)

    def test_analyze_resonance_bits_only_returns_no_fabricated_f0(self):
        """Bits-only callers get f0=None; the retired heuristic is labeled."""
        mon = ResonanceMonitor()
        out = mon.analyze_resonance({"z_bits": [1] * 5, "y_bits": [1] * 16,
                                     "x_bits": [1] * 8})
        self.assertIsNone(out["f0"])
        self.assertIsNone(out["f0_spectral_radius_hz"])
        self.assertAlmostEqual(out["f0_legacy_estimate"], 38.0, delta=1e-6)
        self.assertEqual(out["state"], "NO_SIGNAL")
        self.assertFalse(out["resonance_detected"])


if __name__ == "__main__":
    unittest.main()
