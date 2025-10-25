"""Statevector utilities for n-qubit systems."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


_COMPLEX = np.complex128


def init_state(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("number of qubits must be >= 1")
    psi = np.zeros((2**n, 1), dtype=_COMPLEX)
    psi[0, 0] = 1.0
    return psi


def _axis_for_qubit(qubit: int, n: int) -> int:
    if qubit < 0 or qubit >= n:
        raise ValueError(f"invalid qubit index {qubit} for n={n}")
    return n - 1 - qubit


def apply_unitary(psi: np.ndarray, U: np.ndarray, targets: Iterable[int], n: int) -> np.ndarray:
    targets = list(targets)
    if not targets:
        raise ValueError("apply_unitary requires at least one target qubit")
    dim = 2 ** n
    if psi.shape != (dim, 1):
        raise ValueError(f"psi must have shape {(dim, 1)}, found {psi.shape}")
    m = len(targets)
    if U.shape != (2**m, 2**m):
        raise ValueError("unitary dimension mismatch for provided targets")

    psi_tensor = psi.reshape((2,) * n)
    axes = [_axis_for_qubit(t, n) for t in targets]
    psi_perm = np.moveaxis(psi_tensor, axes, range(m))
    psi_block = psi_perm.reshape(2**m, -1)
    updated = (U @ psi_block).reshape((2,) * n)
    restored = np.moveaxis(updated, range(m), axes)
    return restored.reshape(dim, 1)


def probs(psi: np.ndarray) -> np.ndarray:
    flat = psi.reshape(-1)
    return np.abs(flat) ** 2


def measure_counts(psi: np.ndarray, shots: int = 4096, seed: int = 424242) -> dict[str, int]:
    dim = psi.shape[0]
    n = int(math.log2(dim))
    if dim != 2**n:
        raise ValueError("psi length must be a power of two")
    distribution = probs(psi)
    distribution = distribution / distribution.sum()
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(dim, size=shots, p=distribution)
    counts: dict[str, int] = {}
    for idx in outcomes:
        bitstring = format(idx, f"0{n}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def to_density_matrix(psi: np.ndarray) -> np.ndarray:
    return psi @ psi.conj().T
