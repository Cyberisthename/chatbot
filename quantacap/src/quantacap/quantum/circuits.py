"""Circuit abstraction supporting state-vector and optional GPU execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .statevector import apply_unitary, init_state, measure_counts, probs


@dataclass
class GateApplication:
    U: object
    targets: Tuple[int, ...]


class Circuit:
    def __init__(
        self,
        n: int,
        *,
        seed: int = 424242,
        use_gpu: bool = False,
        dtype: str = "complex128",
    ):
        if n < 1:
            raise ValueError("Circuit requires at least one qubit")
        self.n = n
        self.seed = seed
        self.use_gpu = use_gpu
        self.psi, self.xp = init_state(n, use_gpu=use_gpu, dtype=dtype)
        self.dtype = dtype
        self._ops: List[GateApplication] = []
        self._dirty = False

    def add(self, U, targets: Iterable[int]) -> None:
        targets_tuple = tuple(targets)
        if not targets_tuple:
            raise ValueError("add() requires at least one target qubit")
        unitary = self.xp.asarray(U, dtype=self.psi.dtype)
        self._ops.append(GateApplication(U=unitary, targets=targets_tuple))
        self._dirty = True

    def _refresh(self) -> None:
        if not self._dirty:
            return
        self.psi, self.xp = init_state(self.n, use_gpu=self.use_gpu, dtype=self.dtype)
        for gate in self._ops:
            self.psi = apply_unitary(self.psi, gate.U, gate.targets, self.n, self.xp)
        self._dirty = False

    def run(self):
        self._refresh()
        return self.psi

    def probs(self):
        self._refresh()
        return probs(self.psi, self.xp)

    def measure(self, shots: int = 4096):
        self._refresh()
        return measure_counts(self.psi, shots=shots, seed=self.seed, xp=self.xp)
