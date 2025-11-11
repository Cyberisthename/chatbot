"""Split-operator method for imaginary-time propagation with projection."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .backend import get_array_module
from .fftlaplacian import apply_kinetic


def imaginary_time_step(
    psi: np.ndarray,
    V: np.ndarray,
    kprop: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Perform one split-operator imaginary-time step.

    Order: exp(-V dt/2) * FFT[exp(-k^2 dt/2)] * IFFT * exp(-V dt/2) then normalize.

    Parameters
    ----------
    psi : complex array
        Wavefunction on the grid.
    V : real array
        Potential energy on the grid.
    kprop : complex array
        Kinetic propagator in k-space exp(-0.5 * k^2 * dt).
    dt : float
        Imaginary-time step.
    """

    xp = get_array_module(psi)

    psi = psi * xp.exp(-0.5 * V * dt)
    psi = apply_kinetic(psi, kprop)
    psi = psi * xp.exp(-0.5 * V * dt)

    norm = xp.sqrt(xp.sum(xp.abs(psi) ** 2))
    psi = psi / norm
    return psi


def project_out(psi: np.ndarray, basis_list: List[np.ndarray]) -> np.ndarray:
    """Remove components along basis states (Gram–Schmidt orthogonalization).

    Parameters
    ----------
    psi : complex array
        Wavefunction to be orthogonalized.
    basis_list : list of complex arrays
        Lower-energy orbitals to project out.
    """

    xp = get_array_module(psi)

    for b in basis_list:
        overlap = xp.sum(xp.conj(b) * psi)
        psi = psi - overlap * b

    norm = xp.sqrt(xp.sum(xp.abs(psi) ** 2))
    psi = psi / norm
    return psi


def compute_energy(psi: np.ndarray, V: np.ndarray, k2: np.ndarray, dx: float) -> float:
    """Compute expectation value of kinetic + potential energy."""

    xp = get_array_module(psi)

    psi_k = xp.fft.fftn(psi, norm="ortho")
    kinetic = 0.5 * xp.sum(xp.abs(psi_k) ** 2 * k2).real

    rho = xp.abs(psi) ** 2
    potential = xp.sum(rho * V).real

    dV = dx ** 3
    return float((kinetic + potential) * dV)


__all__ = ["imaginary_time_step", "project_out", "compute_energy"]
