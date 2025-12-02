"""Deterministic convergent graph primitive (G-graph)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = ["GGraph"]


@dataclass
class GGraph:
    n: int = 4096
    out_degree: int = 3
    gamma: float = 0.87
    seed: int = 424242

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.out_degree <= 0:
            raise ValueError("out_degree must be positive")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0,1)")
        rng = np.random.default_rng(self.seed)
        self._edges: List[List[int]] = []
        for node in range(self.n):
            remaining = self.n - node - 1
            if remaining <= 0:
                self._edges.append([])
                continue
            degree = min(self.out_degree, remaining)
            targets = rng.choice(np.arange(node + 1, self.n), size=degree, replace=False)
            self._edges.append(sorted(int(t) for t in targets))

    def influence(self, seed_id: str) -> tuple[float, float]:
        digest = hashlib.sha256(seed_id.encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "big") % self.n
        weights = np.zeros(self.n, dtype=np.float64)
        weights[start] = 1.0
        for node in range(start, self.n):
            w = weights[node]
            if w == 0.0:
                continue
            for tgt in self._edges[node]:
                weights[tgt] += w * self.gamma
        even_sum = float(weights[::2].sum())
        odd_sum = float(weights[1::2].sum())
        scale = 0.5
        eta_a = 0.5 * (math.tanh(scale * even_sum) + 1.0)
        eta_b = 0.5 * (math.tanh(scale * odd_sum) + 1.0)
        return (float(np.clip(eta_a, 0.0, 1.0)), float(np.clip(eta_b, 0.0, 1.0)))

    def summary(self) -> dict:
        seeds = ["demo.ybit", "quantacap", "bell", "fringe"]
        return {
            "n": self.n,
            "out_degree": self.out_degree,
            "gamma": self.gamma,
            "influences": {sid: self.influence(sid) for sid in seeds},
        }
