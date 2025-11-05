"""FFT-based kinetic propagator for imaginary/real-time evolution."""
from __future__ import annotations

import numpy as np

from .backend import get_array_module


def kinetic_propagator(k2: np.ndarray, dt: float) -> np.ndarray:
    """Build exponential kinetic-energy propagator exp(-0.5 * k^2 * dt)."""

    xp = get_array_module(k2)
    return xp.exp(-0.5 * k2 * dt)


def apply_kinetic(psi: np.ndarray, kprop: np.ndarray) -> np.ndarray:
    """Apply kinetic propagator in k-space via FFT."""

    xp = get_array_module(psi)
    psi_k = xp.fft.fftn(psi, norm="ortho")
    psi_k *= kprop
    return xp.fft.ifftn(psi_k, norm="ortho")


__all__ = ["kinetic_propagator", "apply_kinetic"]
