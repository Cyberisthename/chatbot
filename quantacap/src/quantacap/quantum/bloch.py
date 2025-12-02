"""Bloch vector utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def bloch_vector(rho_1q: np.ndarray) -> Tuple[float, float, float]:
    if rho_1q.shape != (2, 2):
        raise ValueError("rho_1q must be 2x2")
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    rx = float(np.real(np.trace(rho_1q @ sx)))
    ry = float(np.real(np.trace(rho_1q @ sy)))
    rz = float(np.real(np.trace(rho_1q @ sz)))
    return rx, ry, rz


def partial_trace_qubit(rho_2q: np.ndarray, keep: int) -> np.ndarray:
    if rho_2q.shape != (4, 4):
        raise ValueError("rho_2q must be 4x4")
    if keep not in (0, 1):
        raise ValueError("keep must be 0 or 1")
    reduced = np.zeros((2, 2), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            bits_i = ((i >> 0) & 1, (i >> 1) & 1)
            bits_j = ((j >> 0) & 1, (j >> 1) & 1)
            if bits_i[1 - keep] == bits_j[1 - keep]:
                row = bits_i[keep]
                col = bits_j[keep]
                reduced[row, col] += rho_2q[i, j]
    return reduced
