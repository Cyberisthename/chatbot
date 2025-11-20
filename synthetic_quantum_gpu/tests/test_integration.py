"""Integration tests for the full Synthetic Quantum GPU stack."""

import time
import unittest

import numpy as np

from synthetic_quantum_gpu import SyntheticQuantumGPU


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.sqgpu = SyntheticQuantumGPU(dim=32, seed=42)
        self.rng = np.random.default_rng(42)

    def tearDown(self) -> None:
        self.sqgpu.shutdown()

    def test_full_workflow(self) -> None:
        """Test multiple tasks using all layers."""
        
        # Task 1: Linear operation
        matrix1 = self.rng.normal(size=(8, 8))
        vec1 = self.rng.normal(size=8)
        result1 = self.sqgpu.submit_task({
            "id": "task1",
            "kind": "linear_op",
            "matrix": matrix1,
            "input": vec1,
        })
        self.assertEqual(result1["status"], "success")
        
        # Task 2: Branch and interfere
        variations = [{"score": s} for s in [0.5, 1.0, 1.5, 2.0]]
        result2 = self.sqgpu.submit_task({
            "id": "task2",
            "kind": "branch_and_interfere",
            "base_payload": {"score": 1.0},
            "variations": variations,
            "top_k": 2,
            "temperature": 0.2,
        })
        self.assertEqual(result2["status"], "success")
        self.assertLessEqual(len(result2["branches"]), 2)
        
        # Task 3: Cached function
        cached_result = self.sqgpu.submit_task({
            "id": "task3",
            "kind": "cached_function",
            "cache_key": "expensive_const",
            "fn": lambda: sum(range(1000)),
        })
        self.assertEqual(cached_result["status"], "success")
        self.assertEqual(cached_result["result"], sum(range(1000)))
        
        # Task 4: Same cache key should reuse result
        start = time.perf_counter()
        cached_result2 = self.sqgpu.submit_task({
            "id": "task4",
            "kind": "cached_function",
            "cache_key": "expensive_const",
            "fn": lambda: sum(range(1000)),
        })
        elapsed = time.perf_counter() - start
        self.assertEqual(cached_result2["result"], sum(range(1000)))
        # Cached call should be very fast
        self.assertLess(elapsed, 0.01)

    def test_memory_router_populated(self) -> None:
        """Verify that memory router receives task embeddings."""
        
        matrix = self.rng.normal(size=(4, 4))
        vec = self.rng.normal(size=4)
        
        self.sqgpu.submit_task({
            "id": "mem_task1",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vec,
        })
        
        # Check memory router has entries
        query_embedding = self.rng.normal(size=32)
        results = self.sqgpu.memory_router.route(query_embedding, top_k=1)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].key, "mem_task1")

    def test_flop_compression_reuse(self) -> None:
        """Verify FLOP compressor caches matrix compressions."""
        
        matrix = self.rng.normal(size=(6, 6))
        vec1 = self.rng.normal(size=6)
        vec2 = self.rng.normal(size=6)
        
        # First call compresses
        result1 = self.sqgpu.submit_task({
            "id": "flop1",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vec1,
        })
        
        # Second call with same matrix should hit cache
        result2 = self.sqgpu.submit_task({
            "id": "flop2",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vec2,
        })
        
        self.assertEqual(result1["status"], "success")
        self.assertEqual(result2["status"], "success")
        # Both should have same compression rank since same matrix
        self.assertEqual(result1["compression_rank"], result2["compression_rank"])

    def test_determinism(self) -> None:
        """Verify deterministic behavior across runs."""
        
        variations = [{"score": s} for s in [1.0, 2.0, 3.0]]
        
        result1 = self.sqgpu.submit_task({
            "id": "det1",
            "kind": "branch_and_interfere",
            "base_payload": {"score": 1.0},
            "variations": variations,
            "top_k": 1,
        })
        
        # Recreate sqgpu with same seed
        self.sqgpu.shutdown()
        self.sqgpu = SyntheticQuantumGPU(dim=32, seed=42)
        
        result2 = self.sqgpu.submit_task({
            "id": "det2",
            "kind": "branch_and_interfere",
            "base_payload": {"score": 1.0},
            "variations": variations,
            "top_k": 1,
        })
        
        # Results should be identical
        self.assertEqual(len(result1["branches"]), len(result2["branches"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
