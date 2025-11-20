"""Mean-field Hartree solver for the helium atom."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..hydrogen.solver import initialize_ground_state, soft_coulomb_potential
from ..numerics.fftlaplacian import kinetic_propagator
from ..numerics.grids import cartesian_grid, k_grid
from ..numerics.poisson import hartree_potential
from ..numerics.splitop import compute_energy, imaginary_time_step, project_out
from ..render import viz3d

SEED = 424242


@dataclass
class HeliumHistory:
    energies: List[float]
    ee_energies: List[float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "total": self.energies,
            "electron_electron": self.ee_energies,
        }


@dataclass
class HeliumResult:
    psi1: NDArray[np.complexfloating]
    psi2: NDArray[np.complexfloating]
    density: NDArray[np.floating]
    potential: NDArray[np.floating]
    hartree: NDArray[np.floating]
    total_energy: float
    ee_energy: float
    history: HeliumHistory


def solve_helium(
    *,
    N: int,
    L: float,
    steps: int,
    dt: float,
    eps: float,
    mix: float,
    spin: str = "singlet",
    tol: float = 1e-5,
    record_every: int = 20,
    seed: int = SEED,
) -> HeliumResult:
    """Solve the helium atom within a mean-field Hartree approximation."""

    if spin not in {"singlet", "triplet"}:
        raise ValueError("spin must be 'singlet' or 'triplet'")
    if not (0.0 < mix < 1.0):
        raise ValueError("mix parameter must be in (0, 1)")

    x, y, z, dx, _, _ = cartesian_grid(N, L)
    r2 = x * x + y * y + z * z

    V_nuclear = soft_coulomb_potential(r2, Z=2.0, eps=eps)
    _, _, _, k2 = k_grid(N, L)
    kprop = kinetic_propagator(k2, dt)

    psi1 = initialize_ground_state(np.sqrt(r2), seed=seed)
    psi2 = initialize_ground_state(np.sqrt(r2), seed=seed + 1)

    density = np.abs(psi1) ** 2 + np.abs(psi2) ** 2
    hartree = hartree_potential(density, L)
    potential = V_nuclear + hartree

    energies: List[float] = []
    ee_history: List[float] = []

    last_energy: Optional[float] = None

    for step in range(steps):
        psi1 = imaginary_time_step(psi1, potential, kprop, dt)
        psi2 = imaginary_time_step(psi2, potential, kprop, dt)

        if spin == "singlet":
            psi2 = psi1.copy()
        else:
            psi2 = project_out(psi2, [psi1])

        new_density = np.abs(psi1) ** 2 + np.abs(psi2) ** 2
        density = (1.0 - mix) * density + mix * new_density

        hartree = hartree_potential(density, L)
        potential = V_nuclear + hartree

        if step % record_every == 0 or step == steps - 1:
            E1 = compute_energy(psi1, potential, k2, dx)
            E2 = compute_energy(psi2, potential, k2, dx)
            ee_energy = 0.5 * float(np.sum(density * hartree) * dx ** 3)
            total_energy = E1 + E2 - ee_energy
            energies.append(total_energy)
            ee_history.append(ee_energy)

            if last_energy is not None and abs(total_energy - last_energy) < tol:
                break
            last_energy = total_energy

    result = HeliumResult(
        psi1=psi1,
        psi2=psi2,
        density=density,
        potential=potential,
        hartree=hartree,
        total_energy=energies[-1],
        ee_energy=ee_history[-1],
        history=HeliumHistory(energies, ee_history),
    )
    return result


def save_artifacts(result: HeliumResult, out_dir: Path, level: float = 0.2) -> None:
    """Persist required helium artifacts."""

    out_dir.mkdir(parents=True, exist_ok=False)

    np.save(out_dir / "psi1.npy", result.psi1.astype(np.complex64))
    np.save(out_dir / "psi2.npy", result.psi2.astype(np.complex64))
    np.save(out_dir / "density.npy", result.density.astype(np.float32))
    np.save(out_dir / "potential.npy", result.potential.astype(np.float32))
    np.save(out_dir / "hartree.npy", result.hartree.astype(np.float32))

    energy_payload = {
        "total_energy": result.total_energy,
        "electron_electron": result.ee_energy,
        "history": result.history.to_dict(),
    }
    (out_dir / "total_energy.json").write_text(json.dumps(energy_payload, indent=2))

    viz3d.save_mips(result.density, out_dir)
    viz3d.marching_cubes_isosurface(result.density, out_dir / "mesh_isosurface.glb", level=level)
    viz3d.make_orbit_mp4(result.density, out_dir / "orbit_spin.mp4")

    ee_payload = {
        "electron_electron_energy": result.ee_energy,
    }
    (out_dir / "ee_energy.json").write_text(json.dumps(ee_payload, indent=2))


__all__ = ["solve_helium", "save_artifacts", "HeliumResult"]
