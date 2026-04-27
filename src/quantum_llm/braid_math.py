"""
Braid Entropy Calculation Logic for JARVIS.
Implements Burau Representation and spectral-radius-based entropy metrics.
"""

import math
from typing import List

import numpy as np


class BraidEntropyCalculator:
    """Calculate braid entropy and information density for reasoning braids."""

    def __init__(self, n_strands: int, t_value: complex = -1.5):
        if n_strands < 2:
            raise ValueError("Braid Group requires at least 2 strands.")
        self.n = n_strands
        self.t = t_value

    def get_generator_matrix(self, i: int, inverse: bool = False) -> np.ndarray:
        """Return the n x n Burau matrix for generator sigma_i (1-indexed)."""
        if not (1 <= i < self.n):
            raise ValueError(f"Generator index {i} out of bounds for {self.n} strands.")

        mat = np.eye(self.n, dtype=complex)
        t = self.t

        if inverse:
            mat[i - 1, i - 1] = 0
            mat[i - 1, i] = 1
            mat[i, i - 1] = 1.0 / t
            mat[i, i] = (t - 1.0) / t
        else:
            mat[i - 1, i - 1] = 1 - t
            mat[i - 1, i] = t
            mat[i, i - 1] = 1
            mat[i, i] = 0
        return mat

    def calculate_braid_matrix(self, braid_word: List[int]) -> np.ndarray:
        """Calculate the product matrix for a braid word."""
        result = np.eye(self.n, dtype=complex)
        for generator in braid_word:
            if generator == 0:
                continue
            is_inverse = generator < 0
            idx = abs(generator)
            mat = self.get_generator_matrix(idx, inverse=is_inverse)
            result = result @ mat
        return result

    def calculate_entropy(self, braid_word: List[int]) -> float:
        """Calculate braid entropy H = log(spectral_radius(M))."""
        if not braid_word:
            return 0.0

        mat = self.calculate_braid_matrix(braid_word)
        try:
            eigenvalues = np.linalg.eigvals(mat)
            spectral_radius = max(np.abs(eigenvalues))
        except np.linalg.LinAlgError:
            spectral_radius = np.linalg.norm(mat, ord=2)
        return float(math.log(max(float(spectral_radius), 1.0)))

    def calculate_normalized_entropy(self, braid_word: List[int]) -> float:
        """Calculate normalized entropy D = H / n."""
        return self.calculate_entropy(braid_word) / self.n

    def classify_novelty(self, braid_word: List[int]) -> str:
        """Classify novelty regime from normalized entropy."""
        density = self.calculate_normalized_entropy(braid_word)
        if density < 0.25:
            return "I: Trivial Extension"
        if density < 0.5:
            return "II: Domain Synthesis"
        return "III: Mechanistic Novelty"


def get_braid_metrics(braid_word: List[int], n_strands: int, t_value: complex = -1.5) -> dict:
    """Return braid entropy metrics used by the hypothesis engine."""
    calc = BraidEntropyCalculator(n_strands=n_strands, t_value=t_value)
    entropy = calc.calculate_entropy(braid_word)
    information_density = calc.calculate_normalized_entropy(braid_word)
    novelty_regime = calc.classify_novelty(braid_word)
    return {
        "braid_entropy": entropy,
        "information_density": information_density,
        "novelty_regime": novelty_regime,
    }
