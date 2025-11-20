import unittest

import numpy as np

from synthetic_quantum_gpu.flop_compression import FlopCompressor


class TestFlopCompression(unittest.TestCase):
    def setUp(self) -> None:
        self.compressor = FlopCompressor(max_cache_items=4)
        self.rng = np.random.default_rng(123)

    def test_linear_compression_accuracy(self) -> None:
        matrix = self.rng.normal(size=(10, 10))
        vector = self.rng.normal(size=10)

        compressed = self.compressor.compress_linear_op(matrix)
        result_compressed = self.compressor.apply_compressed_op(compressed, vector)
        result_exact = matrix @ vector

        np.testing.assert_allclose(result_compressed, result_exact, rtol=1e-2, atol=1e-4)

    def test_cached_function(self) -> None:
        calls = {"count": 0}

        def slow_fn() -> int:
            calls["count"] += 1
            return 42

        first = self.compressor.cached_function("key", slow_fn)
        second = self.compressor.cached_function("key", slow_fn)

        self.assertEqual(first, 42)
        self.assertEqual(second, 42)
        self.assertEqual(calls["count"], 1)

    def test_cache_eviction(self) -> None:
        for i in range(5):
            matrix = self.rng.normal(size=(4, 4))
            self.compressor.compress_linear_op(matrix)
        stats = self.compressor.get_cache_stats()
        self.assertLessEqual(stats["compression_cache_size"], 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
