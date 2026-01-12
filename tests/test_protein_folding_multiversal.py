"""Unit tests for multiversal protein computer.

Uses only Python standard library unittest (no pytest dependency).
"""

import math
import unittest
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import src...` works when tests are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiversal.multiversal_protein_computer import MultiversalProteinComputer
from src.multiversal.protein_folding_engine import ProteinFoldingEngine


class TestMultiversalProteinComputer(unittest.TestCase):
    def test_single_universe_energy_is_finite(self):
        engine = ProteinFoldingEngine(artifacts_dir="./protein_folding_artifacts_test")
        structure = engine.initialize_extended_chain("ACDEFGHIK", seed=123)
        e = engine.energy(structure)
        self.assertTrue(math.isfinite(e))

    def test_single_universe_anneal_improves(self):
        engine = ProteinFoldingEngine(artifacts_dir="./protein_folding_artifacts_test")
        initial = engine.initialize_extended_chain("ACDEFGHIK", seed=123)
        e0 = engine.energy(initial)

        result = engine.metropolis_anneal(initial, steps=500, t_start=2.0, t_end=0.2, seed=123, log_every=250)

        self.assertTrue(math.isfinite(result["best_energy"]))
        self.assertLessEqual(result["best_energy"], e0 + 1e-6)

    def test_multiversal_runs_parallel_universes(self):
        computer = MultiversalProteinComputer(artifacts_dir="./protein_folding_artifacts_test")

        result = computer.fold_multiversal(
            "ACDEFGHIK",
            n_universes=3,
            steps_per_universe=400,
            base_seed=100,
            save_artifacts=False,
        )

        self.assertEqual(result.n_universes, 3)
        self.assertIsNotNone(result.best_overall)
        self.assertTrue(math.isfinite(result.best_overall.best_energy))
        self.assertEqual(len(result.universes), 3)

    def test_multiversal_finds_best_overall(self):
        computer = MultiversalProteinComputer(artifacts_dir="./protein_folding_artifacts_test")

        result = computer.fold_multiversal(
            "ACDEFGH",
            n_universes=4,
            steps_per_universe=300,
            base_seed=200,
            save_artifacts=False,
        )

        # Best overall should be the minimum energy
        energies = [u.best_energy for u in result.universes]
        self.assertEqual(result.best_overall.best_energy, min(energies))

    def test_multiversal_statistics(self):
        computer = MultiversalProteinComputer(artifacts_dir="./protein_folding_artifacts_test")

        result = computer.fold_multiversal(
            "ACDEFG",
            n_universes=4,
            steps_per_universe=200,
            base_seed=300,
            save_artifacts=False,
        )

        # Check statistics
        self.assertTrue(math.isfinite(result.energy_mean))
        self.assertTrue(math.isfinite(result.energy_std))
        self.assertGreaterEqual(result.energy_std, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
