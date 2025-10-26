"""Backend selection utilities for circuit construction."""
from __future__ import annotations

from typing import Literal

from .circuits import Circuit
from .mps import MPSCircuit

BackendName = Literal["statevector", "mps"]


def create_circuit(
    n: int,
    *,
    backend: BackendName = "statevector",
    seed: int = 424242,
    use_gpu: bool = False,
    dtype: str = "complex128",
    chi: int = 16,
):
    if backend == "mps":
        return MPSCircuit(n, chi=chi, seed=seed)
    return Circuit(n, seed=seed, use_gpu=use_gpu, dtype=dtype)
