"""Core helpers for manipulating Quion++ states."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


EXPECTED_DIM = 5


def normalize(psi: Iterable[complex], *, dtype: str | np.dtype = "complex128") -> np.ndarray:
    """Return a normalised copy of the 5-component state vector."""
    arr = np.asarray(list(psi), dtype=dtype)
    if arr.shape != (EXPECTED_DIM,):
        raise ValueError(f"quion state must have {EXPECTED_DIM} amplitudes")
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("cannot normalise a zero Quion state")
    return arr / norm


def mags_phases(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return magnitude-squared and phase vectors for a state."""
    arr = np.asarray(psi, dtype=np.complex128).reshape(EXPECTED_DIM)
    mags = np.abs(arr) ** 2
    total = float(np.sum(mags))
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        mags = mags / total if total > 0 else np.full_like(mags, 1.0 / EXPECTED_DIM)
    phases = np.angle(arr)
    return mags, phases


def stack_re_im(psi: np.ndarray) -> np.ndarray:
    """Stack real and imaginary parts into a 10D real vector."""
    arr = np.asarray(psi, dtype=np.complex128).reshape(EXPECTED_DIM)
    return np.concatenate([arr.real, arr.imag])


def pca2_project(vecs: np.ndarray) -> np.ndarray:
    """Project stacked vectors (T×10) down to 2 dimensions using PCA."""
    if vecs.ndim != 2:
        raise ValueError("expected a 2D array of stacked vectors")
    if vecs.shape[1] != EXPECTED_DIM * 2:
        raise ValueError("stacked vectors must have length 10")
    if len(vecs) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    X = vecs - vecs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2].T
    return X @ basis


def entropy_from_mags(mags: np.ndarray) -> float:
    """Shannon entropy for magnitude distribution."""
    mags = np.asarray(mags, dtype=np.float64)
    mags = np.clip(mags, 1e-12, 1.0)
    return float(-np.sum(mags * np.log(mags)))


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return |⟨a|b⟩|^2 for two normalised states."""
    return float(np.abs(np.vdot(a, b)) ** 2)
