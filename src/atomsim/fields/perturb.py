"""External field perturbations (Stark, Zeeman) and fine-structure effects."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from ..numerics.backend import get_array_module
from ..numerics.fftlaplacian import kinetic_propagator
from ..numerics.grids import cartesian_grid, k_grid
from ..numerics.splitop import compute_energy, imaginary_time_step
from ..render import viz3d

ALPHA = 1.0 / 137.035999084  # fine-structure constant (atomic units)


@dataclass
class FieldResult:
    psi: NDArray[np.complexfloating]
    density: NDArray[np.floating]
    energy: float
    shifts: Dict[str, float]


def _volume_element(dx: float) -> float:
    return dx ** 3


def _expectation(psi: NDArray[np.complexfloating], observable: NDArray[np.complexfloating], dx: float) -> float:
    xp = get_array_module(psi)
    value = xp.sum(xp.conj(psi) * observable) * _volume_element(dx)
    return float(value.real)


def _dipole_moment(psi: NDArray[np.complexfloating], x, y, z, dx: float) -> Tuple[float, float, float]:
    xp = get_array_module(psi)
    rho = xp.abs(psi) ** 2
    dV = _volume_element(dx)
    mx = float(xp.sum(rho * x) * dV)
    my = float(xp.sum(rho * y) * dV)
    mz = float(xp.sum(rho * z) * dV)
    return mx, my, mz


def _orbital_momentum(
    psi: NDArray[np.complexfloating],
    x,
    y,
    z,
    kx,
    ky,
    kz,
    dx: float,
) -> Tuple[float, float, float]:
    xp = get_array_module(psi)
    psi_k = xp.fft.fftn(psi, norm="ortho")

    px_psi = xp.fft.ifftn(1j * kx * psi_k, norm="ortho")
    py_psi = xp.fft.ifftn(1j * ky * psi_k, norm="ortho")
    pz_psi = xp.fft.ifftn(1j * kz * psi_k, norm="ortho")

    Lx = _expectation(psi, y * pz_psi - z * py_psi, dx)
    Ly = _expectation(psi, z * px_psi - x * pz_psi, dx)
    Lz = _expectation(psi, x * py_psi - y * px_psi, dx)
    return Lx, Ly, Lz


def stark_shift(
    psi_initial: NDArray[np.complexfloating],
    V_base: NDArray[np.floating],
    *,
    N: int,
    L: float,
    Ez: float,
    steps: int,
    dt: float,
) -> FieldResult:
    """Simulate Stark effect using imaginary-time relaxation under an added field."""

    x, y, z, dx, _, _ = cartesian_grid(N, L)
    _, _, _, k2 = k_grid(N, L)
    kprop = kinetic_propagator(k2, dt)

    V_stark = V_base - Ez * z
    psi = psi_initial.copy()
    for _ in range(steps):
        psi = imaginary_time_step(psi, V_stark, kprop, dt)

    energy_new = compute_energy(psi, V_stark, k2, dx)
    energy_base = compute_energy(psi_initial, V_base, k2, dx)
    mx, my, mz = _dipole_moment(psi_initial, x, y, z, dx)
    linear_shift = -Ez * mz

    shifts = {
        "stark_shift_numeric": energy_new - energy_base,
        "stark_shift_linear": linear_shift,
        "Ez": Ez,
    }

    return FieldResult(psi=psi, density=np.abs(psi) ** 2, energy=energy_new, shifts=shifts)


def zeeman_shift(
    psi_initial: NDArray[np.complexfloating],
    V_base: NDArray[np.floating],
    *,
    N: int,
    L: float,
    Bz: float,
    steps: int,
    dt: float,
) -> FieldResult:
    """Compute Zeeman splitting using orbital angular momentum expectation."""

    x, y, z, dx, _, _ = cartesian_grid(N, L)
    kx, ky, kz, k2 = k_grid(N, L)
    kprop = kinetic_propagator(k2, dt)

    mu_B = 0.5  # atomic units
    Lx, Ly, Lz = _orbital_momentum(psi_initial, x, y, z, kx, ky, kz, dx)
    linear_shift = mu_B * Bz * Lz

    V_zeeman = V_base - mu_B * Bz * (x * ky - y * kx)

    psi = psi_initial.copy()
    for _ in range(steps):
        psi = imaginary_time_step(psi, V_zeeman, kprop, dt)

    energy_new = compute_energy(psi, V_zeeman, k2, dx)
    energy_base = compute_energy(psi_initial, V_base, k2, dx)

    shifts = {
        "zeeman_shift_numeric": energy_new - energy_base,
        "zeeman_shift_linear": linear_shift,
        "Bz": Bz,
        "Lz_expectation": Lz,
    }

    return FieldResult(psi=psi, density=np.abs(psi) ** 2, energy=energy_new, shifts=shifts)


def fine_structure(
    psi: NDArray[np.complexfloating],
    *,
    Z: float,
    n: int,
    l: int,
    j: float,
    dx: float,
) -> Dict[str, float]:
    """Estimate Darwin and spin-orbit corrections for hydrogen-like atoms."""

    xp = get_array_module(psi)
    center = tuple(s // 2 for s in psi.shape)
    psi0 = float(abs(psi[center]) ** 2)

    darwin = (math.pi * (Z * ALPHA) ** 4 / (2 * n ** 3)) * psi0 * dx ** 3

    if l == 0:
        spin_orbit = 0.0
    else:
        energy_n = -0.5 * Z ** 2 / (n ** 2)
        spin_orbit = energy_n * (Z * ALPHA) ** 2 * (1.0 / (j + 0.5) - 1.0 / (l + 0.5))

    return {
        "darwin_term": float(darwin),
        "spin_orbit": float(spin_orbit),
    }


def save_field_artifacts(result: FieldResult, out_dir: Path, reference_density: NDArray[np.floating]) -> None:
    """Persist field perturbation results and diagnostic plots."""

    out_dir.mkdir(parents=True, exist_ok=False)

    np.save(out_dir / "psi.npy", result.psi.astype(np.complex64))
    np.save(out_dir / "density.npy", result.density.astype(np.float32))

    payload = {
        "final_energy": result.energy,
        **result.shifts,
    }
    (out_dir / "shifts.json").write_text(json.dumps(payload, indent=2))

    viz3d.save_mips(result.density, out_dir)
    viz3d.density_comparison(reference_density, result.density, out_dir)


__all__ = [
    "FieldResult",
    "stark_shift",
    "zeeman_shift",
    "fine_structure",
    "save_field_artifacts",
]
