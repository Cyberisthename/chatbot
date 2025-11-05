"""Split-operator FFT solvers for the hydrogen atom."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import special

from ..numerics.backend import asnumpy, get_array_module
from ..numerics.fftlaplacian import kinetic_propagator
from ..numerics.grids import cartesian_grid, k_grid, radial_coordinates
from ..numerics.poisson import hartree_potential
from ..numerics.splitop import compute_energy, imaginary_time_step, project_out
from ..render import viz3d

SEED = 424242


@dataclass
class SolverHistory:
    energies: List[float]
    norms: List[float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"energies": self.energies, "norms": self.norms}


@dataclass
class SolverResult:
    psi: NDArray[np.complexfloating]
    energy: float
    potential: NDArray[np.floating]
    density: NDArray[np.floating]
    history: SolverHistory


def soft_coulomb_potential(
    r2: NDArray[np.floating],
    Z: float,
    eps: float,
) -> NDArray[np.floating]:
    """Softened Coulomb potential -Z/sqrt(r^2 + eps^2)."""

    return -Z / np.sqrt(r2 + eps * eps)


def initialize_ground_state(
    r: NDArray[np.floating],
    seed: int = SEED,
) -> NDArray[np.complexfloating]:
    """Deterministic Gaussian initial guess."""

    rng = np.random.default_rng(seed)
    sigma = np.max(r) / 6.0
    psi = np.exp(-(r ** 2) / (2.0 * sigma * sigma))
    psi *= (1.0 + 0.05 * rng.normal(size=psi.shape))
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def hydrogenic_radial(n: int, l: int, r: NDArray[np.floating], Z: float = 1.0):
    """Analytic hydrogenic radial function (unnormalised)."""

    rho = 2.0 * Z * r / n
    prefactor = (2.0 * Z / n) ** 3 * math.sqrt(math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    laguerre = special.genlaguerre(n - l - 1, 2 * l + 1)(rho)
    radial = prefactor * np.exp(-rho / 2.0) * rho ** l * laguerre
    return radial


def initialize_excited_state(
    n: int,
    l: int,
    m: int,
    r: NDArray[np.floating],
    theta: NDArray[np.floating],
    phi: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Hydrogenic initial guess from spherical harmonics."""

    radial = hydrogenic_radial(n, l, r)
    Y_lm = special.sph_harm(m, l, phi, theta)
    psi0 = np.nan_to_num(radial * Y_lm)
    norm = np.linalg.norm(psi0)
    if norm == 0:
        raise ValueError("Initial excited state has zero norm")
    return (psi0 / norm).astype(np.complex128)


def solve_ground(
    N: int,
    L: float,
    Z: float,
    steps: int,
    dt: float,
    eps: float,
    record_every: int = 10,
    seed: int = SEED,
) -> SolverResult:
    """Imaginary-time propagation to find the ground-state orbital."""

    xp = np
    x, y, z, dx, _, _ = cartesian_grid(N, L, xp=xp)
    r = np.sqrt(x * x + y * y + z * z)
    r2 = r * r

    V = soft_coulomb_potential(r2, Z=Z, eps=eps)
    kx, ky, kz, k2 = k_grid(N, L, xp=xp)
    kprop = kinetic_propagator(k2, dt)

    psi = initialize_ground_state(r, seed=seed)

    energies: List[float] = []
    norms: List[float] = []

    for step in range(steps):
        psi = imaginary_time_step(psi, V, kprop, dt)
        if step % record_every == 0 or step == steps - 1:
            E = compute_energy(psi, V, k2, dx)
            energies.append(E)
            norms.append(float(np.linalg.norm(psi)))

    density = np.abs(psi) ** 2
    result = SolverResult(
        psi=psi,
        energy=energies[-1],
        potential=V,
        density=density,
        history=SolverHistory(energies, norms),
    )
    return result


def solve_excited(
    n: int,
    l: int,
    m: int,
    *,
    N: int,
    L: float,
    Z: float,
    steps: int,
    dt: float,
    eps: float,
    lower_states: Optional[Iterable[np.ndarray]] = None,
    record_every: int = 10,
    seed: int = SEED,
) -> SolverResult:
    """Solve for an excited hydrogen orbital using projection."""

    xp = np
    x, y, z, dx, _, _ = cartesian_grid(N, L, xp=xp)
    r, theta, phi = radial_coordinates(x, y, z)
    r2 = r * r

    V = soft_coulomb_potential(r2, Z=Z, eps=eps)
    _, _, _, k2 = k_grid(N, L, xp=xp)
    kprop = kinetic_propagator(k2, dt)

    psi = initialize_excited_state(n, l, m, r, theta, phi)
    template = psi.copy()

    basis: List[np.ndarray]
    if lower_states is None:
        ground = solve_ground(N=N, L=L, Z=Z, steps=steps // 4, dt=dt, eps=eps, seed=seed)
        basis = [ground.psi]
    else:
        basis = [np.asarray(b) for b in lower_states]

    energies: List[float] = []
    norms: List[float] = []

    for step in range(steps):
        psi = imaginary_time_step(psi, V, kprop, dt)
        psi = project_out(psi, basis)

        # Preserve nodal structure by aligning with template sign
        overlap = np.vdot(template, psi)
        if overlap.real < 0:
            psi = -psi

        if step % record_every == 0 or step == steps - 1:
            E = compute_energy(psi, V, k2, dx)
            energies.append(E)
            norms.append(float(np.linalg.norm(psi)))

    density = np.abs(psi) ** 2
    result = SolverResult(
        psi=psi,
        energy=energies[-1],
        potential=V,
        density=density,
        history=SolverHistory(energies, norms),
    )
    return result


def save_artifacts(
    result: SolverResult,
    out_dir: Path,
    *,
    analytic_label: Optional[str] = None,
    level: float = 0.2,
    box_length: float,
) -> None:
    """Persist NumPy arrays and visualizations as required by the ticket."""

    out_dir.mkdir(parents=True, exist_ok=False)

    density = result.density.astype(np.float32)
    np.save(out_dir / "psi.npy", result.psi.astype(np.complex64))
    np.save(out_dir / "density.npy", density)
    np.save(out_dir / "potential.npy", result.potential.astype(np.float32))

    energy_payload = {
        "final": result.energy,
        "history": result.history.to_dict(),
    }
    (out_dir / "energy.json").write_text(json.dumps(energy_payload, indent=2))

    viz3d.save_mips(density, out_dir)
    if analytic_label == "1s":
        viz3d.radial_profile(density, out_dir, box_length=box_length)
    viz3d.marching_cubes_isosurface(density, out_dir / "mesh_isosurface.glb", level=level)
    viz3d.make_orbit_mp4(density, out_dir / "orbit_spin.mp4")


__all__ = [
    "SolverResult",
    "soft_coulomb_potential",
    "solve_ground",
    "solve_excited",
    "save_artifacts",
]
