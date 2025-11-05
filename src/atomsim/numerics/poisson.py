"""Hartree potential solver using FFT-based Poisson solver."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .backend import get_array_module
from .grids import k_grid


def hartree_potential(rho: np.ndarray, L: float, eps: float = 1e-12) -> np.ndarray:
    """Solve Poisson equation for Hartree potential of density ``rho``.

    The solver assumes periodic boundary conditions in a cubic domain of side
    length ``L``. The zero-frequency component is forced to zero to avoid the
    singularity at k=0 (equivalent to subtracting the mean potential).
    """

    xp = get_array_module(rho)
    N = rho.shape[0]

    rho_k = xp.fft.fftn(rho, norm="ortho")
    _, _, _, k2 = k_grid(N, L, xp=xp)

    inv_k2 = xp.zeros_like(k2)
    xp.divide(1.0, k2, out=inv_k2, where=k2 >= eps)
    factor = 4.0 * math.pi
    V_k = factor * rho_k * inv_k2
    V = xp.fft.ifftn(V_k, norm="ortho")

    V = V.real
    V = V - xp.mean(V)
    return V


__all__ = ["hartree_potential"]
