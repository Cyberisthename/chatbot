"""CHSH experiment with Y-bit and G-graph modulation."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.experiments.chsh import _correlation_from_probs, _measurement_probs
from quantacap.primitives.ggraph import GGraph
from quantacap.primitives.ybit import YBit
from quantacap.primitives.zbit import ZBit
from quantacap.quantum.backend import create_circuit
from quantacap.quantum.gates import CNOT, H
from quantacap.quantum.noise import apply_channel_rho, depolarizing
from quantacap.quantum.xp import to_numpy

Angles = Dict[str, float]
PairMap = Dict[str, Tuple[str, str]]

SETTINGS: Tuple[Angles, PairMap] = (
    {
        "A": 0.0,
        "A'": math.pi / 2.0,
        "B": math.pi / 4.0,
        "B'": -math.pi / 4.0,
    },
    {
        "AB": ("A", "B"),
        "AB'": ("A", "B'"),
        "A'B": ("A'", "B"),
        "A'B'": ("A'", "B'"),
    },
)


def _local_amplitudes(psi: np.ndarray, qubit: int) -> Tuple[complex, complex]:
    if psi.ndim != 1:
        psi = psi.reshape(-1)
    alpha = 0.0 + 0.0j
    beta = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        bit = (idx >> qubit) & 1
        if bit:
            beta += amp
        else:
            alpha += amp
    return alpha, beta


def _rz_matrix(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128
    )


def _apply_rz_frame(rho: np.ndarray, delta_a: float, delta_b: float) -> np.ndarray:
    rz_a = _rz_matrix(delta_a)
    rz_b = _rz_matrix(delta_b)
    U = np.kron(rz_a, rz_b)
    return U @ rho @ U.conj().T


def _mix_distribution(
    probs: np.ndarray,
    target: np.ndarray,
    blend: float,
) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    mixed = (1.0 - blend) * probs + blend * target
    mixed = np.clip(mixed, 0.0, None)
    total = float(np.sum(mixed))
    if total <= 0.0:
        return np.full_like(mixed, 0.25)
    return mixed / total


def run_chsh_y(
    *,
    shots: int = 50000,
    depol: float = 0.0,
    seed: int = 424242,
    seed_id: str = "demo.ybit",
    lam: float = 0.85,
    eps: float = 0.02,
    delta: float = 0.03,
    graph_nodes: int = 4096,
    graph_out: int = 3,
    graph_gamma: float = 0.87,
    backend: str = "statevector",
    use_gpu: bool = False,
    dtype: str = "complex128",
    chi: int = 32,
    adapter_id: str | None = None,
) -> Dict[str, object]:
    angles_base, pairs = SETTINGS
    circuit = create_circuit(2, backend=backend, seed=seed, use_gpu=use_gpu, dtype=dtype, chi=chi)
    xp = getattr(circuit, "xp", None)
    circuit.add(H(xp=xp, dtype=dtype), [0])
    circuit.add(CNOT(xp=xp, dtype=dtype), [0, 1])
    psi = to_numpy(xp, circuit.run()).reshape(-1)
    rho = psi[:, None] @ psi.conj()[None, :]

    rng = np.random.default_rng(seed)
    z0 = ZBit(seed=int(rng.integers(0, 2**32)))
    z1 = ZBit(seed=int(rng.integers(0, 2**32)))
    y0 = YBit(_local_amplitudes(psi, 0), z0, lam=lam, eps_phase=eps)
    y1 = YBit(_local_amplitudes(psi, 1), z1, lam=lam, eps_phase=eps)
    phase_a = y0.phase_nudge()
    phase_b = y1.phase_nudge()

    rho = _apply_rz_frame(rho, phase_a, phase_b)

    if depol:
        for qubit in (0, 1):
            kraus_ops = depolarizing(depol, [qubit], n=2)
            rho = apply_channel_rho(rho, kraus_ops)

    graph = GGraph(n=graph_nodes, out_degree=graph_out, gamma=graph_gamma, seed=seed)
    eta_a, eta_b = graph.influence(seed_id)

    shift_a = delta * ((eta_a - 0.5) + (y0.adjusted_prob_1() - 0.5))
    shift_b = delta * ((eta_b - 0.5) + (y1.adjusted_prob_1() - 0.5))

    angles = {
        "A": angles_base["A"] + shift_a,
        "A'": angles_base["A'"] + shift_a,
        "B": angles_base["B"] + shift_b,
        "B'": angles_base["B'"] + shift_b,
    }

    blend = 1.0 - ((y0.lam + y1.lam) / 2.0)

    rng_counts = np.random.default_rng(seed)
    counts: Dict[str, Dict[str, int]] = {}
    expectations: Dict[str, float] = {}

    for label, (akey, bkey) in pairs.items():
        probs = _measurement_probs(rho, angles[akey], angles[bkey])
        target = np.array(
            [
                (1.0 - y0.adjusted_prob_1()) * (1.0 - y1.adjusted_prob_1()),
                (1.0 - y0.adjusted_prob_1()) * y1.adjusted_prob_1(),
                y0.adjusted_prob_1() * (1.0 - y1.adjusted_prob_1()),
                y0.adjusted_prob_1() * y1.adjusted_prob_1(),
            ],
            dtype=float,
        )
        probs = _mix_distribution(probs, target, blend)
        expectations[label] = _correlation_from_probs(probs)
        draws = rng_counts.choice(4, size=shots, p=probs)
        bucket: Dict[str, int] = {}
        for outcome in draws:
            bitstring = format(int(outcome), "02b")
            bucket[bitstring] = bucket.get(bitstring, 0) + 1
        counts[label] = bucket

    S = expectations["AB"] + expectations["AB'"] + expectations["A'B"] - expectations["A'B'"]

    result = {
        "S": float(S),
        "terms": expectations,
        "counts": counts,
        "shots_per_setting": shots,
        "noise": {"depol": float(depol)},
        "angles": angles,
        "mod": {
            "phiA": float(phase_a),
            "phiB": float(phase_b),
            "etaA": float(eta_a),
            "etaB": float(eta_b),
            "lam": float(lam),
            "eps": float(eps),
            "delta": float(delta),
        },
        "seed": int(seed),
        "seed_id": seed_id,
    }

    if adapter_id:
        os.makedirs("artifacts", exist_ok=True)
        artifact_path = os.path.join("artifacts", f"{adapter_id}.json")
        with open(artifact_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        create_adapter(adapter_id, data=result, meta={"kind": "chsh_y"})

    return result
