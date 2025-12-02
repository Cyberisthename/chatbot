"""CHSH Bell-inequality experiment helpers."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from quantacap.quantum.backend import create_circuit
from quantacap.quantum.gates import CNOT, H, RY
from quantacap.quantum.noise import apply_channel_rho, depolarizing
from quantacap.quantum.xp import to_numpy


def _measurement_unitary(angle: float) -> np.ndarray:
    gate = RY(-angle)
    return np.asarray(gate)


def _measurement_probs(rho: np.ndarray, angle_a: float, angle_b: float) -> np.ndarray:
    Ua = _measurement_unitary(angle_a)
    Ub = _measurement_unitary(angle_b)
    U = np.kron(Ua, Ub)
    rotated = U @ rho @ U.conj().T
    probs = np.real_if_close(np.diag(rotated))
    probs = np.clip(probs, 0.0, None)
    return probs / probs.sum()


def _correlation_from_probs(probs: np.ndarray) -> float:
    expectation = 0.0
    for idx, prob in enumerate(probs):
        parity = -1 if (idx.bit_count() % 2) else 1
        expectation += parity * float(prob)
    return expectation


def run_chsh(
    *,
    shots: int = 50000,
    depol: float = 0.0,
    seed: int = 424242,
    backend: str = "statevector",
    use_gpu: bool = False,
    dtype: str = "complex128",
    chi: int = 32,
) -> Dict[str, object]:
    circuit = create_circuit(2, backend=backend, seed=seed, use_gpu=use_gpu, dtype=dtype, chi=chi)
    xp = getattr(circuit, "xp", None)
    circuit.add(H(xp=xp), [0])
    circuit.add(CNOT(xp=xp), [0, 1])
    psi = to_numpy(xp, circuit.run())
    rho = psi @ psi.conj().T

    if depol:
        for qubit in (0, 1):
            kraus_ops = depolarizing(depol, [qubit], n=2)
            rho = apply_channel_rho(rho, kraus_ops)

    angles: Dict[str, float] = {
        "A": 0.0,
        "A'": math.pi / 2,
        "B": math.pi / 4,
        "B'": -math.pi / 4,
    }

    pairs: Dict[str, Tuple[str, str]] = {
        "AB": ("A", "B"),
        "AB'": ("A", "B'"),
        "A'B": ("A'", "B"),
        "A'B'": ("A'", "B'"),
    }

    rng = np.random.default_rng(seed)
    counts: Dict[str, Dict[str, int]] = {}
    expectations: Dict[str, float] = {}

    for label, (a_key, b_key) in pairs.items():
        probs = _measurement_probs(rho, angles[a_key], angles[b_key])
        expectations[label] = _correlation_from_probs(probs)
        outcomes = rng.choice(4, size=shots, p=probs)
        bucket: Dict[str, int] = {}
        for idx in outcomes:
            bitstring = format(int(idx), "02b")
            bucket[bitstring] = bucket.get(bitstring, 0) + 1
        counts[label] = bucket

    S = expectations["AB"] + expectations["AB'"] + expectations["A'B"] - expectations["A'B'"]
    return {
        "S": float(S),
        "terms": expectations,
        "counts": counts,
        "shots_per_setting": shots,
        "noise": {"depol": float(depol)},
    }
