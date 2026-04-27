"""Unit tests for braid entropy metrics."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantum_llm.braid_math import BraidEntropyCalculator, get_braid_metrics


class TestBraidEntropyCalculator(unittest.TestCase):
    def test_linear_word_is_trivial(self):
        calc = BraidEntropyCalculator(n_strands=2)
        metrics = get_braid_metrics([1], n_strands=2)
        self.assertGreaterEqual(metrics["braid_entropy"], 0.0)
        self.assertIn("I:", metrics["novelty_regime"])

    def test_nontrivial_braid_has_higher_density(self):
        calc = BraidEntropyCalculator(n_strands=4)
        trivial = calc.calculate_normalized_entropy([1])
        novel = calc.calculate_normalized_entropy([1, 3, 2] * 6)
        self.assertGreater(novel, trivial)
        self.assertIn("III:", calc.classify_novelty([1, 3, 2] * 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
