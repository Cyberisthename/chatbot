"""Synthetic atom density experiments."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple

import numpy as np

from quantacap.core.adapter_store import create_adapter, load_adapter
from quantacap.quantum.xp import get_xp, to_numpy


def generate_atom_state(
    *,
    n: int,
    L: float,
    sigma: float,
    use_gpu: bool = False,
    dtype: str = "complex128",
):
    xp = get_xp(use_gpu)
    N = 1 << n
    grid = np.linspace(-L, L, N)
    dt = xp.complex128 if dtype == "complex128" else xp.complex64
    psi = xp.exp(-(xp.asarray(grid) ** 2) / (2.0 * sigma**2)).astype(dt)
    psi = psi.reshape(N, 1)
    norm = xp.sqrt(xp.sum(xp.abs(psi) ** 2))
    psi = psi / norm
    density = to_numpy(xp, xp.abs(psi.reshape(-1)) ** 2)
    return grid, psi, density


def run_atom1d(
    *,
    n: int,
    L: float,
    sigma: float,
    adapter_id: str,
    use_gpu: bool = False,
    dtype: str = "complex128",
) -> Dict[str, object]:
    grid, psi, density = generate_atom_state(n=n, L=L, sigma=sigma, use_gpu=use_gpu, dtype=dtype)
    os.makedirs("artifacts", exist_ok=True)
    artifact_path = os.path.join("artifacts", f"atom1d_{adapter_id}.json")
    payload = {
        "id": adapter_id,
        "n": n,
        "L": L,
        "sigma": sigma,
        "grid": grid.tolist(),
        "density": density.tolist(),
    }
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    amps = to_numpy(get_xp(use_gpu), psi.reshape(-1))
    backend_info = {
        "type": "statevector",
        "device": "gpu" if use_gpu else "cpu",
        "chi": None,
    }
    create_adapter(
        adapter_id,
        state={
            "n": n,
            "dtype": dtype,
            "amps": amps,
            "probs": density.tolist(),
            "backend": backend_info,
        },
        meta={"L": L, "sigma": sigma, "dtype": dtype},
    )
    return {"artifact": artifact_path, "density": density.tolist(), "grid": grid.tolist()}


def _phase_coupling_from_pi(pi_adapter_id: str | None) -> Tuple[float, float]:
    """Extract a phase shift and weight from a π-phase experiment adapter."""

    if not pi_adapter_id:
        return 0.0, 1.0
    try:
        record = load_adapter(pi_adapter_id)
    except FileNotFoundError:
        return 0.0, 1.0

    data = record.get("data", {}) if isinstance(record, dict) else {}
    entropy = None
    for key in ("entropy_final", "entropy_mean", "entropy"):
        if key in data:
            try:
                entropy = float(data[key])
                break
            except (TypeError, ValueError):
                continue
    if entropy is None and isinstance(data, dict):
        entropy = float(data.get("metrics", {}).get("entropy", 0.0)) if isinstance(data.get("metrics"), dict) else 0.0

    entropy = float(entropy or 0.0)
    weight = 1.0 / (1.0 + entropy)
    phase_shift = math.pi * (1.0 - math.tanh(entropy))
    return phase_shift, weight


def run_atom2d_transition(
    *,
    n: int,
    L: float,
    sigma_primary: float,
    sigma_secondary: float,
    separation: float,
    adapter_id: str,
    pi_adapter_id: str | None = None,
    use_gpu: bool = False,
    dtype: str = "complex128",
) -> Dict[str, object]:
    """Simulate two coupled wells forming an interference pattern.

    The function extends :func:`run_atom1d` to a 2D lattice, applies two
    Gaussian wells separated by ``separation`` and introduces a relative phase
    driven by a π-phase adapter.  The output density is flattened to save into
    JSON while preserving replay fidelity through the adapter payload.
    """

    if n % 2 != 0:
        raise ValueError("n must be even to map to a square lattice")

    xp = get_xp(use_gpu)
    dim = 1 << (n // 2)
    grid = np.linspace(-L, L, dim)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    phase_shift, weight = _phase_coupling_from_pi(pi_adapter_id)

    primary = np.exp(-(((X + separation / 2) ** 2 + Y**2) / (2.0 * sigma_primary**2)))
    secondary = np.exp(-(((X - separation / 2) ** 2 + Y**2) / (2.0 * sigma_secondary**2)))
    psi = primary + weight * np.exp(1j * phase_shift) * secondary
    psi = psi.astype(np.complex128)
    norm = np.linalg.norm(psi)
    if not norm:
        raise ValueError("wavefunction norm is zero")
    psi /= norm
    density = np.abs(psi) ** 2

    fringe_visibility = float(density.max() - density.min())

    artifact_path = os.path.join("artifacts", f"atom2d_{adapter_id}.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    payload = {
        "id": adapter_id,
        "n": n,
        "L": L,
        "sigma_primary": sigma_primary,
        "sigma_secondary": sigma_secondary,
        "separation": separation,
        "phase_shift": phase_shift,
        "weight": weight,
        "grid": grid.tolist(),
        "density": density.tolist(),
        "fringe_visibility": fringe_visibility,
    }
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    backend_info = {"type": "statevector", "device": "gpu" if use_gpu else "cpu", "chi": None}
    amps = psi.reshape(-1)
    create_adapter(
        adapter_id,
        state={
            "n": n,
            "dtype": dtype,
            "amps": amps,
            "probs": density.reshape(-1).tolist(),
            "backend": backend_info,
        },
        meta={
            "L": L,
            "sigma_primary": sigma_primary,
            "sigma_secondary": sigma_secondary,
            "separation": separation,
            "phase_shift": phase_shift,
            "weight": weight,
        },
        data={"fringe_visibility": fringe_visibility},
    )

    return {
        "artifact": artifact_path,
        "density": density.tolist(),
        "grid": grid.tolist(),
        "phase_shift": phase_shift,
        "weight": weight,
        "fringe_visibility": fringe_visibility,
    }
