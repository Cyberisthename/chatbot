"""Minimal Matrix Product State utilities for low-entanglement circuits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .statevector import measure_counts as sv_measure_counts
from .xp import get_xp, to_numpy


@dataclass
class GateApplication:
    U: object
    targets: Tuple[int, ...]


def init_mps(n: int, chi: int, xp):
    if n < 1:
        raise ValueError("MPS requires at least one qubit")
    tensors: List[object] = []
    dt = xp.complex128
    for idx in range(n):
        left = 1 if idx == 0 else 1
        right = 1 if idx == n - 1 else 1
        tensor = xp.zeros((left, 2, right), dtype=dt)
        tensor[0, 0, 0] = 1.0
        tensors.append(tensor)
    return tensors


def _apply_gate_on_physical(site, gate, xp):
    updated = xp.tensordot(gate, site, axes=[1, 1])  # (2, left, right)
    return xp.transpose(updated, (1, 0, 2))


def apply_1q_mps(mps, gate, target, xp):
    mps[target] = _apply_gate_on_physical(mps[target], gate, xp)


def _svd_truncate(matrix: np.ndarray, chi: int):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = min(len(s), chi)
    return u[:, :keep], s[:keep], vh[:keep, :]


def apply_2q_mps(mps, gate, t0, t1, xp, chi: int):
    if t1 != t0 + 1:
        raise ValueError("This minimal MPS implementation expects nearest neighbours")
    left_tensor = mps[t0]
    right_tensor = mps[t1]
    theta = xp.tensordot(left_tensor, right_tensor, axes=[2, 0])  # (Dl,2,2,Dr)
    Dl, _, _, Dr = theta.shape
    theta = xp.reshape(theta, (Dl, 4, Dr))
    theta = xp.reshape(theta, (Dl * Dr, 4)).T  # (4, Dl*Dr)
    updated = xp.matmul(gate, theta)
    updated = xp.reshape(updated, (4, Dl, Dr))
    updated = xp.transpose(updated, (1, 0, 2))  # (Dl,4,Dr)
    updated = xp.reshape(updated, (Dl, 2, 2, Dr))
    matrix = xp.reshape(updated, (Dl * 2, 2 * Dr))
    matrix_np = to_numpy(xp, matrix)
    u, s, vh = _svd_truncate(matrix_np, chi)
    kept = u.shape[1]
    xp_u = xp.asarray(u.reshape(Dl, 2, kept))
    xp_s = xp.asarray(s)
    xp_vh = xp.asarray(vh)
    left_new = xp_u
    right_new = xp.asarray((xp_s[:, None] * xp_vh)).reshape(kept, 2, Dr)
    mps[t0] = left_new
    mps[t1] = right_new


def mps_to_statevector(mps, xp):
    tensor = xp.reshape(mps[0], (2, mps[0].shape[2]))
    for site in mps[1:]:
        tensor = xp.tensordot(tensor, site, axes=[-1, 0])  # (...,2, bond)
        tensor = xp.reshape(tensor, (-1, site.shape[2]))
    return tensor.reshape(-1, 1)


def probs_mps(mps, xp):
    psi = mps_to_statevector(mps, xp)
    amps = to_numpy(xp, psi.reshape(-1))
    return np.abs(amps) ** 2


class MPSCircuit:
    def __init__(self, n: int, *, chi: int = 16, seed: int = 424242):
        self.n = n
        self.seed = seed
        self.chi = chi
        self.xp = get_xp(False)
        self.mps = init_mps(n, chi, self.xp)
        self._ops: List[GateApplication] = []
        self._dirty = False

    def add(self, U, targets: Iterable[int]):
        targets_tuple = tuple(targets)
        if len(targets_tuple) not in (1, 2):
            raise ValueError("MPS backend supports 1 or 2 qubit gates")
        unitary = self.xp.asarray(U)
        self._ops.append(GateApplication(unitary, targets_tuple))
        self._dirty = True

    def _apply(self):
        self.mps = init_mps(self.n, self.chi, self.xp)
        for op in self._ops:
            if len(op.targets) == 1:
                apply_1q_mps(self.mps, op.U, op.targets[0], self.xp)
            else:
                apply_2q_mps(self.mps, op.U, op.targets[0], op.targets[1], self.xp, self.chi)
        self._dirty = False

    def _ensure(self):
        if self._dirty:
            self._apply()

    def run(self):
        self._ensure()
        return mps_to_statevector(self.mps, self.xp)

    def probs(self):
        self._ensure()
        return probs_mps(self.mps, self.xp)

    def measure(self, shots: int = 4096):
        self._ensure()
        return sv_measure_counts(mps_to_statevector(self.mps, self.xp), shots=shots, seed=self.seed)
