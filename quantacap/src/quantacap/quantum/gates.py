"""Elementary quantum gates for Quantacap."""
from __future__ import annotations

import math
import numpy as np


_COMPLEX = np.complex128


def kron_n(*mats: np.ndarray) -> np.ndarray:
    """Kronecker product of the provided matrices."""
    if not mats:
        raise ValueError("kron_n requires at least one matrix")
    out = np.array([[1.0]], dtype=_COMPLEX)
    for mat in mats:
        out = np.kron(out, np.asarray(mat, dtype=_COMPLEX))
    return out


def I() -> np.ndarray:
    return np.eye(2, dtype=_COMPLEX)


def X() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=_COMPLEX)


def Z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=_COMPLEX)


def H() -> np.ndarray:
    return (1 / math.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=_COMPLEX)


def RZ(theta: float) -> np.ndarray:
    phase = theta / 2.0
    return np.array(
        [
            [np.exp(-1j * phase), 0.0],
            [0.0, np.exp(1j * phase)],
        ],
        dtype=_COMPLEX,
    )


def CNOT() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=_COMPLEX,
    )
