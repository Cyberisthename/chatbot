import unittest

import numpy as np

from synthetic_quantum_gpu.infinite_router import InfiniteMemoryRouter


class TestInfiniteMemoryRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = InfiniteMemoryRouter(dim=8, max_entries=100)
        self.rng = np.random.default_rng(42)

    def test_add_and_route(self) -> None:
        emb1 = self.rng.normal(size=8)
        emb2 = self.rng.normal(size=8)
        self.router.add_entry("key1", emb1, {"val": 1})
        self.router.add_entry("key2", emb2, {"val": 2})

        results = self.router.route(emb1, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].key, "key1")

    def test_snapshot_restore(self) -> None:
        emb = self.rng.normal(size=8)
        self.router.add_entry("key", emb, {"val": 123})
        snap = self.router.snapshot()

        router2 = InfiniteMemoryRouter(dim=8)
        router2.load_snapshot(snap)

        results = router2.route(emb, top_k=1)
        self.assertEqual(results[0].key, "key")
        self.assertEqual(results[0].payload["val"], 123)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
