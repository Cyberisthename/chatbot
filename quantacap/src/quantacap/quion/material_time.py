"""Entropy-based material time ticking for Quion++ animations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .state import entropy_from_mags, mags_phases


@dataclass
class TickResult:
    commit: bool
    entropy: float
    delta_entropy: float


class MaterialClock:
    """Decide whether to commit a frame based on entropy changes."""

    def __init__(self, tau: float = 1e-4, guard: bool = False):
        self.tau = float(tau)
        self.guard = bool(guard)
        self._prev_state: Optional[np.ndarray] = None
        self._prev_entropy: Optional[float] = None
        self._prev_v: Optional[float] = None
        self.violations: int = 0

    def observe(self, psi: np.ndarray) -> TickResult:
        mags, _ = mags_phases(psi)
        entropy = entropy_from_mags(mags)
        if self._prev_entropy is None:
            self._prev_state = psi.copy()
            self._prev_entropy = entropy
            self._prev_v = 0.0
            return TickResult(True, entropy, 0.0)

        delta_entropy = entropy - float(self._prev_entropy)
        commit = abs(delta_entropy) > self.tau

        if self.guard and self._prev_state is not None:
            diff = float(np.linalg.norm(psi - self._prev_state) ** 2)
            v_curr = abs(delta_entropy) + diff
            if self._prev_v is not None and v_curr > self._prev_v + 1e-9:
                self.violations += 1
                commit = False
            else:
                self._prev_v = v_curr
        else:
            self._prev_v = abs(delta_entropy)

        if commit:
            self._prev_state = psi.copy()
            self._prev_entropy = entropy

        return TickResult(commit, entropy, delta_entropy)

    def status(self) -> Tuple[float, Optional[float]]:
        return (float(self.tau), self._prev_entropy)
