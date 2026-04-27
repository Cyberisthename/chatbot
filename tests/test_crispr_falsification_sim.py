"""Tests for the CRISPR falsification simulation."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.crispr_falsification_sim import CrisprFalsificationSimulator


class TestCrisprFalsificationSimulation(unittest.TestCase):
    def setUp(self):
        self.simulator = CrisprFalsificationSimulator(n_targets=48, n_guides=5, inference_cycles=4, base_seed=11)

    def test_single_trial_improves_off_target_load(self):
        trial = self.simulator.run_trial(seed=11)
        self.assertLess(trial.arm_c.mean_off_target_load, trial.arm_a.mean_off_target_load)
        self.assertGreater(trial.arm_c.effective_fidelity_ratio, trial.arm_a.effective_fidelity_ratio)
        self.assertTrue(trial.arm_c_quantum.stable_constructive_interference)

    def test_aggregate_support_fraction_is_positive(self):
        report = self.simulator.run_trials(n_trials=6)
        self.assertGreater(report.mean_relative_reduction, 0.0)
        self.assertGreater(report.mean_fidelity_ratio_gain, 0.0)
        self.assertGreater(report.support_fraction, 0.5)
        self.assertGreaterEqual(report.arm_c_stable_fraction, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
