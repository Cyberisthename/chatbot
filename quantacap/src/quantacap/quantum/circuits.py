"""Lightweight circuit runner built on the statevector utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .statevector import apply_unitary, init_state, measure_counts as sv_measure_counts, probs as sv_probs


@dataclass
class GateApplication:
    U: np.ndarray
    targets: Tuple[int, ...]


class Circuit:
    def __init__(self, n: int, seed: int = 424242):
        if n < 1:
            raise ValueError("Circuit requires at least one qubit")
        self.n = n
        self.seed = seed
        self._ops: List[GateApplication] = []
        self._psi = init_state(n)
        self._dirty = True

    def add(self, U: np.ndarray, targets: Iterable[int]) -> None:
        targets_tuple = tuple(targets)
        if not targets_tuple:
            raise ValueError("add() requires at least one target qubit")
        self._ops.append(GateApplication(U=np.asarray(U, dtype=np.complex128), targets=targets_tuple))
        self._dirty = True

    def _refresh(self) -> None:
        if not self._dirty:
            return
        self._psi = init_state(self.n)
        for gate in self._ops:
            self._psi = apply_unitary(self._psi, gate.U, gate.targets, self.n)
        self._dirty = False

    def run(self) -> np.ndarray:
        self._refresh()
        return self._psi.copy()

    def probs(self) -> np.ndarray:
        self._refresh()
        return sv_probs(self._psi.copy())

    def measure(self, shots: int = 4096) -> dict[str, int]:
        self._refresh()
        return sv_measure_counts(self._psi, shots=shots, seed=self.seed)
