"""Z-bit primitive with a forbidden band between 1 and 2."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ZBit"]


_EPS = 1e-12


def _reflect(value: float) -> float:
    """Ensure ``value`` stays outside the forbidden band [1, 2]."""

    if 1.0 <= value <= 2.0:
        if value <= 1.5:
            return 1.0 - _EPS
        return 2.0 + _EPS
    return value


@dataclass
class ZBit:
    """Scalar living on the real line excluding the interval [1, 2]."""

    seed: int = 424242
    side: str = "auto"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        side = self.side
        if side not in {"auto", "left", "right"}:
            raise ValueError("side must be 'auto', 'left', or 'right'")
        if side == "auto":
            side = "left" if rng.random() < 0.5 else "right"
        sample = abs(float(rng.normal()))
        if side == "left":
            value = 1.0 - sample - _EPS
        else:
            value = 2.0 + sample + _EPS
        self._value = _reflect(value)

    def value(self) -> float:
        return float(self._value)

    def bias_sigma(self, k: float = 1.0) -> float:
        center = 1.5
        delta = abs(self._value - center)
        signed = math.copysign(math.log1p(delta), self._value)
        exponent = -k * signed
        sigma = 1.0 / (1.0 + math.exp(exponent))
        return float(min(max(sigma, 0.0), 1.0))

    def phase(self) -> float:
        phi = math.fmod(abs(self._value), 2.0 * math.pi)
        if phi < 0:
            phi += 2.0 * math.pi
        return float(phi)
