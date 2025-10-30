"""Noise-collapse sweep for π-phase rotations."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run


def _sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    steps: int,
    schedule: str,
) -> np.ndarray:
    count = max(2, steps)
    if schedule == "log":
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError("log schedule requires positive sigma bounds")
        lower, upper = (min(sigma_min, sigma_max), max(sigma_min, sigma_max))
        seq = np.geomspace(lower, upper, count)
        if sigma_max < sigma_min:
            seq = seq[::-1]
        return seq
    if schedule != "linear":
        raise ValueError(f"Unknown schedule '{schedule}'")
    return np.linspace(sigma_min, sigma_max, count)


def _detect_plateaus(entropy: list[float], epsilon: float) -> list[Dict[str, float]]:
    plateaus: list[Dict[str, float]] = []
    if not entropy:
        return plateaus
    start = 0
    for idx in range(1, len(entropy)):
        if abs(entropy[idx] - entropy[idx - 1]) > epsilon:
            if idx - 1 >= start:
                window = entropy[start:idx]
                plateaus.append(
                    {
                        "start": start,
                        "end": idx - 1,
                        "mean_entropy": float(sum(window) / len(window)),
                    }
                )
            start = idx
    if start < len(entropy):
        window = entropy[start:]
        plateaus.append(
            {
                "start": start,
                "end": len(entropy) - 1,
                "mean_entropy": float(sum(window) / len(window)),
            }
        )
    return plateaus


def run_pi_noise_scan(
    *,
    sigma_max: float,
    steps: int,
    rotations: int = 1000,
    sigma_min: float = 0.0,
    entropy_threshold: float = 0.05,
    entropy_bins: int = 64,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
    schedule: str = "linear",
    detect_steps: bool = True,
    track_entropy: bool = True,
    plateau_epsilon: float = 1e-3,
) -> Dict[str, object]:
    if sigma_max < 0 or sigma_min < 0:
        raise ValueError("sigma values must be non-negative")
    sigmas = _sigma_schedule(sigma_min, sigma_max, steps, schedule)
    rng = np.random.default_rng(seed)
    coherence = []
    t0 = time.perf_counter()
    entropy: list[float] = []
    entropy_deltas: list[float] = []
    two_pi = 2.0 * math.pi
    phase_state = np.full(rotations, math.pi, dtype=float)

    for sigma in sigmas:
        phase_state = (phase_state + rng.normal(0.0, sigma, size=rotations)) % two_pi
        phases = phase_state.copy()
        value = abs(np.mean(np.exp(1j * phases)))
        coherence.append(float(value))

        if track_entropy:
            counts, _ = np.histogram(phases, bins=entropy_bins, range=(0.0, two_pi))
            total = counts.sum()
            if total == 0:
                entropy.append(0.0)
            else:
                probs = counts / total
                nonzero = probs[probs > 0]
                entropy.append(float(-np.sum(nonzero * np.log(nonzero))))

            if len(entropy) >= 2:
                entropy_deltas.append(float(entropy[-1] - entropy[-2]))
            else:
                entropy_deltas.append(0.0)
        else:
            entropy.append(0.0)
            entropy_deltas.append(0.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    discrete_steps = (
        [
            idx
            for idx, delta in enumerate(entropy_deltas)
            if abs(delta) >= entropy_threshold and idx > 0
        ]
        if detect_steps and track_entropy
        else []
    )

    plateaus = _detect_plateaus(entropy, plateau_epsilon) if track_entropy else []

    result = {
        "sigma": sigmas.tolist(),
        "coherence": coherence,
        "entropy": entropy,
        "entropy_delta": entropy_deltas,
        "entropy_steps": discrete_steps,
        "entropy_plateaus": plateaus,
        "threshold_index": _first_below(coherence, 0.5),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "entropy_threshold": entropy_threshold,
        "seed": seed,
        "schedule": schedule,
    }
    artifact_path = artifact_path or "artifacts/pi_noise_scan.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    adapter_id = adapter_id or f"pi.noise.{seed}"
    create_adapter(
        adapter_id,
        data=result,
        meta={
            "experiment": "pi_noise",
            "rotations": rotations,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "entropy_threshold": entropy_threshold,
        },
    )
    log_quantum_run(
        "pi.noise",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"coherence": coherence[-1], "entropy": entropy[-1], "S": None},
        delta_v=None,
    )
    return result


def _first_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values):
        if value < threshold:
            return idx
    return None


def run_pi_entropy_collapse(
    *,
    kappa: float,
    sigma_min: float = 1e-9,
    sigma_max: float = 1e-6,
    stages: int = 25,
    stage_length: int = 128,
    entropy_threshold: float = 1e-3,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    """Scan the coupling experiment while gradually increasing phase noise.

    The routine couples two π-locked oscillators (as in ``run_pi_coupling``) and
    injects Gaussian noise with a slowly increasing standard deviation.  The
    coherence and entropy traces are sampled at the end of each stage and
    analysed for discrete entropy drops which would signal a synthetic entropy
    collapse event.
    """

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma bounds must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")

    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), max(2, stages))
    rng = np.random.default_rng(seed)

    phi = np.array([math.pi, math.pi + 1e-6], dtype=float)
    history: list[np.ndarray] = []
    entropy_trace: list[float] = []
    coherence_trace: list[float] = []

    two_pi = 2.0 * math.pi

    t0 = time.perf_counter()
    for sigma in sigmas:
        for _ in range(max(1, stage_length)):
            diff = phi[1] - phi[0]
            phi[0] += kappa * diff
            phi[1] -= kappa * diff
            phi += rng.normal(0.0, sigma, size=2)
            phi = np.mod(phi, two_pi)
            history.append(phi.copy())

        block = np.array(history[-stage_length:], dtype=float)
        coherence_value = abs(np.mean(np.exp(1j * block))) if block.size else 0.0
        coherence_trace.append(float(coherence_value))

        phases = block.reshape(-1)
        counts, _ = np.histogram(phases, bins=64, range=(0.0, two_pi))
        probs = counts / counts.sum() if counts.sum() else counts
        nz = probs[probs > 0]
        entropy_value = float(-np.sum(nz * np.log(nz))) if nz.size else 0.0
        entropy_trace.append(entropy_value)

    entropy_deltas = [0.0]
    for idx in range(1, len(entropy_trace)):
        entropy_deltas.append(float(entropy_trace[idx] - entropy_trace[idx - 1]))

    discrete_steps = [
        idx
        for idx, delta in enumerate(entropy_deltas)
        if abs(delta) >= entropy_threshold and idx > 0
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    artifact_path = artifact_path or "artifacts/pi_entropy_collapse.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "kappa": kappa,
        "sigma": sigmas.tolist(),
        "coherence": coherence_trace,
        "entropy": entropy_trace,
        "entropy_delta": entropy_deltas,
        "entropy_steps": discrete_steps,
        "stage_length": stage_length,
        "seed": seed,
    }

    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    adapter_id = adapter_id or f"pi.collapse.{seed}"
    create_adapter(
        adapter_id,
        data=result,
        meta={
            "experiment": "pi_entropy_collapse",
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "entropy_threshold": entropy_threshold,
        },
    )
    log_quantum_run(
        "pi.collapse",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"coherence": coherence_trace[-1], "entropy": entropy_trace[-1], "S": None},
        delta_v=None,
    )
    return result
