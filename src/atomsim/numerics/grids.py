"""Grid construction helpers for real and reciprocal space."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .backend import get_array_module

Array = Tuple[np.ndarray, np.ndarray, np.ndarray]


def cartesian_grid(
    N: int,
    L: float,
    *,
    xp=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Return real-space coordinates for a cubic grid.

    Parameters
    ----------
    N : int
        Number of samples along each dimension.
    L : float
        Physical side length of the cubic box (in Bohr radii).
    xp : module, optional
        Array module (NumPy or CuPy). Defaults to NumPy.
    """

    if xp is None:
        xp = np

    dx = L / N
    axis = xp.linspace(-0.5 * L, 0.5 * L - dx, N, dtype=float)
    x, y, z = xp.meshgrid(axis, axis, axis, indexing="ij")
    return x, y, z, dx, dx, dx


def k_grid(
    N: int,
    L: float,
    *,
    xp=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reciprocal-space wavevectors and their squared magnitude."""

    if xp is None:
        xp = np

    dx = L / N
    k_axis = 2.0 * math.pi * xp.fft.fftfreq(N, d=dx)
    kx, ky, kz = xp.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def radial_coordinates(x, y, z):
    """Return spherical coordinates (r, theta, phi) from Cartesian grids."""

    xp = get_array_module(x)
    r = xp.sqrt(x * x + y * y + z * z)
    # Avoid division by zero at the origin by clipping
    with xp.errstate(invalid="ignore"):
        theta = xp.arccos(xp.clip(xp.where(r == 0, 1.0, z / xp.where(r == 0, 1.0, r)), -1.0, 1.0))
    phi = xp.arctan2(y, x)
    return r, theta, phi


__all__ = ["cartesian_grid", "k_grid", "radial_coordinates"]
