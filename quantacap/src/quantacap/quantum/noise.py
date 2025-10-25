"""Simple noise channels implemented via Kraus operators."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .statevector import apply_unitary

_COMPLEX = np.complex128


def _expand_operator(U: np.ndarray, targets: Iterable[int], n: int) -> np.ndarray:
    targets = list(targets)
    if not targets:
        raise ValueError("targets must be provided")
    dim = 2**n
    out = np.zeros((dim, dim), dtype=_COMPLEX)
    for basis in range(dim):
        col = np.zeros((dim, 1), dtype=_COMPLEX)
        col[basis, 0] = 1.0
        transformed = apply_unitary(col, U, targets, n)
        out[:, basis] = transformed[:, 0]
    return out


def depolarizing(p: float, targets: Iterable[int], n: int) -> List[np.ndarray]:
    if p < 0 or p > 1:
        raise ValueError("depolarizing probability must be in [0,1]")
    targets = list(targets)
    if len(targets) != 1:
        raise ValueError("depolarizing currently supports exactly one target qubit")
    t = targets[0]
    sqrt_one = np.sqrt(1 - 3 * p / 4)
    sqrt_p = np.sqrt(p / 4)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=_COMPLEX)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=_COMPLEX)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=_COMPLEX)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=_COMPLEX)
    base_ops = [sqrt_one * I, sqrt_p * X, sqrt_p * Y, sqrt_p * Z]
    return [_expand_operator(op, [t], n) for op in base_ops]


def phase_damp(gamma: float, targets: Iterable[int], n: int) -> List[np.ndarray]:
    if gamma < 0 or gamma > 1:
        raise ValueError("phase damping parameter must be in [0,1]")
    targets = list(targets)
    if len(targets) != 1:
        raise ValueError("phase damping currently supports exactly one target qubit")
    t = targets[0]
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=_COMPLEX)
    k1 = np.array([[0.0, 0.0], [0.0, np.sqrt(gamma)]], dtype=_COMPLEX)
    return [_expand_operator(k, [t], n) for k in (k0, k1)]


def apply_channel_rho(rho: np.ndarray, kraus_ops: Iterable[np.ndarray]) -> np.ndarray:
    updated = np.zeros_like(rho, dtype=_COMPLEX)
    for K in kraus_ops:
        updated = updated + K @ rho @ K.conj().T
    return updated


def apply_channel_state(psi: np.ndarray, kraus_ops: Iterable[np.ndarray], seed: int = 424242) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ops = list(kraus_ops)
    weights = np.array([np.linalg.norm(K @ psi) for K in ops], dtype=float)
    probs = weights**2
    if probs.sum() == 0:
        return psi
    probs = probs / probs.sum()
    choice = rng.choice(len(ops), p=probs)
    new_state = ops[choice] @ psi
    norm = np.linalg.norm(new_state)
    if norm == 0:
        return psi
    return new_state / norm
