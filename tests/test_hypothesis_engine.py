"""Unit tests for the refined autonomous hypothesis engine."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research import AutonomousHypothesisEngine


class TestAutonomousHypothesisEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AutonomousHypothesisEngine(inference_cycles=4)
        self.data_path = Path(__file__).resolve().parents[1] / "demos" / "modern_research_observations.json"
        self.observations = self.engine.load_observations_from_json(self.data_path)

    def test_load_observations(self):
        self.assertGreaterEqual(len(self.observations), 4)
        self.assertTrue(all(obs.entities for obs in self.observations))

    def test_generate_candidates(self):
        self.engine.ingest_observations(self.observations)
        candidates = self.engine.generate_candidate_hypotheses()
        self.assertGreaterEqual(len(candidates), len(self.observations))
        self.assertTrue(any(candidate.candidate_type == "cross_domain_analogy" for candidate in candidates))

    def test_real_quantum_feedback_trace(self):
        self.engine.ingest_observations(self.observations)
        candidate = self.engine.generate_candidate_hypotheses()[0]
        metrics = self.engine.collect_quantum_metrics(candidate)
        self.assertEqual(metrics.source, "transformer_forward")
        self.assertEqual(len(metrics.cycle_metrics), self.engine.inference_cycles)
        self.assertGreater(metrics.interference, 0.0)
        self.assertGreaterEqual(metrics.braid_entropy, 0.0)
        self.assertIsNotNone(metrics.stable_interference)

    def test_proposals_include_braid_and_falsification(self):
        proposals = self.engine.propose_hypotheses(self.observations, top_k=3)
        self.assertGreaterEqual(len(proposals), 1)
        top = proposals[0]
        self.assertGreater(top.evaluation.overall_score, 0.0)
        self.assertGreater(top.evaluation.quantum_metrics.interference, 0.0)
        self.assertGreaterEqual(top.evaluation.braid_novelty, 0.0)
        self.assertIn("braid", top.falsification_plan.required_data.lower())
        self.assertTrue(top.falsification_plan.controls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
