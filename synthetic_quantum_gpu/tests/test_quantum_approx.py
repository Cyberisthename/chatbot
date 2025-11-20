import unittest

import numpy as np

from synthetic_quantum_gpu.quantum_approx import QuantumApproximator
from synthetic_quantum_gpu.types import BranchState


class TestQuantumApprox(unittest.TestCase):
    def setUp(self) -> None:
        self.approx = QuantumApproximator(seed=100)

    def test_spawn_branches(self) -> None:
        base = BranchState(id="root", amplitude=complex(1.0, 0.0), payload={"x": 1})
        variations = [{"y": 2}, {"y": 3}]
        branches = self.approx.spawn_branches(base, variations)

        self.assertEqual(len(branches), 2)
        self.assertEqual(branches[0].payload["x"], 1)
        self.assertEqual(branches[0].payload["y"], 2)
        self.assertEqual(branches[1].payload["y"], 3)

    def test_interfere_normalization(self) -> None:
        branches = [
            BranchState(id="b1", amplitude=complex(0.5, 0.0), payload={"score": 1.0}),
            BranchState(id="b2", amplitude=complex(0.5, 0.0), payload={"score": 2.0}),
        ]

        result = self.approx.interfere(branches, lambda b: b.payload["score"], temperature=0.1)

        total_prob = sum(np.abs(b.amplitude) ** 2 for b in result.branches)
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_collapse_determinism(self) -> None:
        branches = [
            BranchState(id="b1", amplitude=complex(0.6, 0.0), payload={"score": 1.0}),
            BranchState(id="b2", amplitude=complex(0.4, 0.0), payload={"score": 0.5}),
        ]
        result = self.approx.interfere(branches, lambda b: b.payload["score"], temperature=0.1)
        collapsed1 = self.approx.collapse(result, lambda b: b.payload["score"], top_k=1)

        self.approx.reseed(100)
        collapsed2 = self.approx.collapse(result, lambda b: b.payload["score"], top_k=1)

        self.assertEqual(collapsed1[0].id, collapsed2[0].id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
