"""Bell-state helpers for Quantacap."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .backend import create_circuit
from .gates import CNOT, H
from .noise import apply_channel_rho, depolarizing, phase_damp
from .xp import to_numpy


def _probabilities_from_density(rho: np.ndarray) -> np.ndarray:
    probs = np.real_if_close(np.diag(rho))
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total == 0:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / total


def bell_counts(
    shots: int = 8192,
    seed: int = 424242,
    noise: Optional[Dict[str, float]] = None,
    *,
    backend: str = "statevector",
    use_gpu: bool = False,
    dtype: str = "complex128",
    chi: int = 32,
) -> Dict[str, int]:
    circuit = create_circuit(2, backend=backend, seed=seed, use_gpu=use_gpu, dtype=dtype, chi=chi)
    circuit.add(H(xp=circuit.xp), [0])
    circuit.add(CNOT(xp=circuit.xp), [0, 1])

    if noise:
        psi = to_numpy(circuit.xp, circuit.run())
        rho = psi @ psi.conj().T
        if noise.get("depol"):
            p = float(noise["depol"])
            for qubit in (0, 1):
                kraus_ops = depolarizing(p, [qubit], n=2)
                rho = apply_channel_rho(rho, kraus_ops)
        if noise.get("phase"):
            gamma = float(noise["phase"])
            for qubit in (0, 1):
                kraus_ops = phase_damp(gamma, [qubit], n=2)
                rho = apply_channel_rho(rho, kraus_ops)
        probs = _probabilities_from_density(rho)
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(4, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for idx in outcomes:
            bitstring = format(idx, "02b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    return circuit.measure(shots=shots)
