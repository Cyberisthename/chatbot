"""Noise rehearsal scan for CHSH experiments."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from quantacap.core.adapter_store import create_adapter, load_adapter
from quantacap.experiments.chsh import _correlation_from_probs, _measurement_probs
from quantacap.quantum.backend import create_circuit
from quantacap.quantum.bloch import bloch_vector, partial_trace_qubit
from quantacap.quantum.gates import CNOT, H, RZ
from quantacap.quantum.noise import apply_channel_rho, depolarizing
from quantacap.quantum.xp import to_numpy

Angles = Dict[str, float]
PairMap = Dict[str, Tuple[str, str]]


@dataclass(frozen=True)
class CHSHSettings:
    angles: Angles
    pairs: PairMap


SETTINGS = CHSHSettings(
    angles={
        "A": 0.0,
        "A'": math.pi / 2.0,
        "B": math.pi / 4.0,
        "B'": -math.pi / 4.0,
    },
    pairs={
        "AB": ("A", "B"),
        "AB'": ("A", "B'"),
        "A'B": ("A'", "B"),
        "A'B'": ("A'", "B'"),
    },
)


def _prepare_bell_density(*, backend: str, use_gpu: bool, dtype: str, chi: int, seed: int) -> np.ndarray:
    circuit = create_circuit(2, backend=backend, seed=seed, use_gpu=use_gpu, dtype=dtype, chi=chi)
    xp = getattr(circuit, "xp", None)
    circuit.add(H(xp=xp, dtype=dtype), [0])
    circuit.add(CNOT(xp=xp, dtype=dtype), [0, 1])
    psi = to_numpy(xp, circuit.run())
    return psi @ psi.conj().T


def _apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    noisy = np.array(rho, copy=True)
    if p <= 0:
        return noisy
    for qubit in (0, 1):
        kraus_ops = depolarizing(p, [qubit], n=2)
        noisy = apply_channel_rho(noisy, kraus_ops)
    return noisy


def _simulate_counts(
    rho: np.ndarray,
    shots: int,
    *,
    seed: int,
    settings: CHSHSettings,
) -> Tuple[float, Dict[str, Dict[str, int]], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    counts: Dict[str, Dict[str, int]] = {}
    expectations: Dict[str, float] = {}
    for label, (akey, bkey) in settings.pairs.items():
        probs = _measurement_probs(rho, settings.angles[akey], settings.angles[bkey])
        expectations[label] = _correlation_from_probs(probs)
        outcomes = rng.choice(4, size=shots, p=probs)
        bucket: Dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(int(outcome), "02b")
            bucket[bitstring] = bucket.get(bitstring, 0) + 1
        counts[label] = bucket
    S = expectations["AB"] + expectations["AB'"] + expectations["A'B"] - expectations["A'B'"]
    return float(S), counts, expectations


def _bloch_payload(rho: np.ndarray) -> Dict[str, Dict[str, float]]:
    reduced0 = partial_trace_qubit(rho, keep=0)
    reduced1 = partial_trace_qubit(rho, keep=1)
    rx0, ry0, rz0 = bloch_vector(reduced0)
    rx1, ry1, rz1 = bloch_vector(reduced1)
    return {
        "q0": {"rx": rx0, "ry": ry0, "rz": rz0},
        "q1": {"rx": rx1, "ry": ry1, "rz": rz1},
    }


def _phase_from_bloch(bloch: Dict[str, float]) -> float:
    rx = float(bloch.get("rx", 0.0))
    ry = float(bloch.get("ry", 0.0))
    if abs(rx) < 1e-12 and abs(ry) < 1e-12:
        return 0.0
    return math.atan2(ry, rx)


def _apply_phase_alignment(
    *,
    phi0: float,
    phi1: float,
    backend: str,
    use_gpu: bool,
    dtype: str,
    chi: int,
    seed: int,
) -> np.ndarray:
    circuit = create_circuit(2, backend=backend, seed=seed, use_gpu=use_gpu, dtype=dtype, chi=chi)
    xp = getattr(circuit, "xp", None)
    circuit.add(H(xp=xp, dtype=dtype), [0])
    circuit.add(CNOT(xp=xp, dtype=dtype), [0, 1])
    circuit.add(RZ(-phi0, xp=xp, dtype=dtype), [0])
    circuit.add(RZ(-phi1, xp=xp, dtype=dtype), [1])
    psi = to_numpy(xp, circuit.run())
    return psi @ psi.conj().T


def _format_id(prefix: str, p: float) -> str:
    return f"{prefix}.p{p:.4f}".rstrip("0").rstrip(".")


def _find_peak(values: Sequence[float]) -> int | None:
    if len(values) < 3:
        return None
    arr = np.asarray(values)
    idx = int(np.argmax(arr))
    if idx == 0 or idx == len(arr) - 1:
        return None
    if arr[idx] <= arr[idx - 1] + 1e-6:
        return None
    if arr[idx] <= arr[idx + 1] + 1e-6:
        return None
    return idx


def run_chsh_scan(
    *,
    pmin: float,
    pmax: float,
    steps: int,
    shots: int,
    adapter_id: str,
    backend: str = "statevector",
    use_gpu: bool = False,
    dtype: str = "complex128",
    chi: int = 32,
    seed: int = 424242,
) -> Dict[str, object]:
    if steps < 2:
        raise ValueError("steps must be >= 2")
    ps = np.linspace(pmin, pmax, steps)
    rho_clean = _prepare_bell_density(backend=backend, use_gpu=use_gpu, dtype=dtype, chi=chi, seed=seed)
    S_clean, clean_counts, expectations_clean = _simulate_counts(
        rho_clean, shots, seed=seed, settings=SETTINGS
    )

    os.makedirs("artifacts", exist_ok=True)

    S_noisy: List[float] = []
    S_rehearsed: List[float] = []
    phases: List[Tuple[float, float]] = []

    for idx, p in enumerate(ps):
        rho_noisy = _apply_depolarizing(rho_clean, float(p))
        bloch = _bloch_payload(rho_noisy)
        S_val, counts, expectations = _simulate_counts(
            rho_noisy, shots, seed=seed + idx + 1, settings=SETTINGS
        )
        noisy_id = _format_id("chsh", float(p))
        create_adapter(
            noisy_id,
            data={
                "p": float(p),
                "shots": shots,
                "S": S_val,
                "counts": counts,
                "angles": SETTINGS.angles,
                "expectations": expectations,
                "bloch": bloch,
            },
            meta={"kind": "chsh_noise"},
        )
        record = load_adapter(noisy_id)
        bloch_data = record["data"].get("bloch", {})
        phi0 = _phase_from_bloch(bloch_data.get("q0", {}))
        phi1 = _phase_from_bloch(bloch_data.get("q1", {}))
        phases.append((phi0, phi1))

        rho_aligned = _apply_phase_alignment(
            phi0=phi0,
            phi1=phi1,
            backend=backend,
            use_gpu=use_gpu,
            dtype=dtype,
            chi=chi,
            seed=seed,
        )
        rho_aligned_noisy = _apply_depolarizing(rho_aligned, float(p))
        S_re, counts_re, expectations_re = _simulate_counts(
            rho_aligned_noisy, shots, seed=seed + steps + idx + 1, settings=SETTINGS
        )
        rehearse_id = _format_id("chsh.rehearsed", float(p))
        create_adapter(
            rehearse_id,
            data={
                "p": float(p),
                "shots": shots,
                "S": S_re,
                "counts": counts_re,
                "phi": [phi0, phi1],
                "expectations": expectations_re,
            },
            meta={"kind": "chsh_rehearsed"},
        )

        S_noisy.append(float(S_val))
        S_rehearsed.append(float(S_re))

    peak_idx = _find_peak(S_rehearsed)
    peak_payload = None
    if peak_idx is not None:
        peak_payload = {"p_star": float(ps[peak_idx]), "S": float(S_rehearsed[peak_idx])}

    result = {
        "id": adapter_id,
        "shots": shots,
        "p": ps.tolist(),
        "S_clean": float(S_clean),
        "S_noisy": S_noisy,
        "S_rehearsed": S_rehearsed,
        "phases": [[float(a), float(b)] for a, b in phases],
        "angles": SETTINGS.angles,
        "expectations_clean": expectations_clean,
        "counts_clean": clean_counts,
        "peak": peak_payload,
    }

    artifact_path = os.path.join("artifacts", f"{adapter_id}.json")
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    create_adapter(adapter_id, data=result, meta={"kind": "chsh_scan"})
    return result
