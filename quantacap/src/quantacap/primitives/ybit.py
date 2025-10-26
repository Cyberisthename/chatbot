"""Y-bit hybrid primitive combining qubit amplitudes and a Z-bit bias."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .zbit import ZBit

__all__ = ["YBit"]


@dataclass
class YBit:
    alpha_beta: Tuple[complex, complex]
    zbit: ZBit
    lam: float = 0.85
    eps_phase: float = 0.02
    bias_k: float = 1.0

    def __post_init__(self) -> None:
        lam = float(np.clip(self.lam, 0.0, 1.0))
        self.lam = lam
        eps = float(abs(self.eps_phase))
        self.eps_phase = eps
        alpha, beta = self.alpha_beta
        norm = abs(alpha) ** 2 + abs(beta) ** 2
        if norm <= 0:
            raise ValueError("alpha_beta must define a non-zero state")
        # normalise to guard against numeric drift
        scale = 1.0 / math.sqrt(norm)
        self._alpha = complex(alpha) * scale
        self._beta = complex(beta) * scale
        self._p1 = float(abs(self._beta) ** 2)
        sigma_b = float(self.zbit.bias_sigma(self.bias_k))
        mixed = lam * self._p1 + (1.0 - lam) * sigma_b
        self._adjusted = float(np.clip(mixed, 0.0, 1.0))

    def adjusted_prob_1(self) -> float:
        return self._adjusted

    def base_prob_1(self) -> float:
        return self._p1

    def phase_nudge(self) -> float:
        return float(self.eps_phase * self.zbit.phase())

    def info(self) -> dict:
        return {
            "alpha": [float(np.real(self._alpha)), float(np.imag(self._alpha))],
            "beta": [float(np.real(self._beta)), float(np.imag(self._beta))],
            "p1_base": self._p1,
            "p1_adjusted": self._adjusted,
            "lam": self.lam,
            "eps_phase": self.eps_phase,
        }
