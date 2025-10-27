"""Synthetic atom 1-D density experiment."""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
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
