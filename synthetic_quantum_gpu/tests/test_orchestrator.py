import unittest

import numpy as np

from synthetic_quantum_gpu import SyntheticQuantumGPU


class TestSyntheticQuantumGPU(unittest.TestCase):
    def setUp(self) -> None:
        self.sqgpu = SyntheticQuantumGPU(dim=16)
        self.rng = np.random.default_rng(123)

    def tearDown(self) -> None:
        self.sqgpu.shutdown()

    def test_linear_op_task(self) -> None:
        matrix = self.rng.normal(size=(4, 4))
        vector = self.rng.normal(size=4)

        result = self.sqgpu.submit_task({
            "id": "lin1",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vector,
        })

        np.testing.assert_allclose(result["result"], matrix @ vector, rtol=1e-2, atol=1e-4)
        self.assertEqual(result["status"], "success")

    def test_branch_and_interfere_task(self) -> None:
        variations = [{"score": v} for v in (1.0, 0.5, 2.0)]

        result = self.sqgpu.submit_task({
            "id": "branch1",
            "kind": "branch_and_interfere",
            "base_payload": {"score": 1.0},
            "variations": variations,
            "top_k": 2,
        })

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["branches"]), 2)

    def test_cached_function_task(self) -> None:
        calls = {"count": 0}

        def expensive() -> int:
            calls["count"] += 1
            return 7

        first = self.sqgpu.submit_task({
            "id": "cache1",
            "kind": "cached_function",
            "cache_key": "const",
            "fn": expensive,
        })
        second = self.sqgpu.submit_task({
            "id": "cache2",
            "kind": "cached_function",
            "cache_key": "const",
            "fn": expensive,
        })

        self.assertEqual(first["result"], 7)
        self.assertEqual(second["result"], 7)
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
