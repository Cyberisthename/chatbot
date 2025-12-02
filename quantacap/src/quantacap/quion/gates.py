"""Reversible gate primitives for Quion++ states."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .state import EXPECTED_DIM


def H5(dtype: str | np.dtype = "complex128") -> np.ndarray:
    """Discrete Fourier transform over 5 components (unitary)."""
    omega = np.exp(2j * np.pi / EXPECTED_DIM)
    mat = np.empty((EXPECTED_DIM, EXPECTED_DIM), dtype=dtype)
    for j in range(EXPECTED_DIM):
        for k in range(EXPECTED_DIM):
            mat[j, k] = omega ** (j * k)
    return mat / np.sqrt(EXPECTED_DIM)


def PHASE5(theta_vec: Iterable[float], dtype: str | np.dtype = "complex128") -> np.ndarray:
    theta = np.asarray(list(theta_vec), dtype=float)
    if theta.size != EXPECTED_DIM:
        raise ValueError("theta vector must have length 5")
    diag = np.exp(1j * theta)
    return np.diag(diag.astype(dtype))


def YG_BIAS(eps: float, vec: Iterable[float], dtype: str | np.dtype = "complex128") -> np.ndarray:
    weights = np.asarray(list(vec), dtype=float)
    if weights.size != EXPECTED_DIM:
        raise ValueError("bias vector must have length 5")
    phases = eps * weights
    return PHASE5(phases, dtype=dtype)


def RANDOM_UNITARY5(seed: int = 424242, dtype: str | np.dtype = "complex128") -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(EXPECTED_DIM, EXPECTED_DIM)) + 1j * rng.normal(size=(EXPECTED_DIM, EXPECTED_DIM))
    q, r = np.linalg.qr(mat)
    diag = np.diag(r)
    phases = np.ones_like(diag)
    nonzero = np.abs(diag) > 1e-12
    phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    U = q * phases
    return U.astype(dtype)


def apply(psi: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return gate @ psi


def invert(gate: np.ndarray) -> np.ndarray:
    return gate.conj().T
