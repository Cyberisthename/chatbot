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
) -> Dict[str, object]:
    if sigma_max < 0 or sigma_min < 0:
        raise ValueError("sigma values must be non-negative")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")
    sigmas = np.linspace(sigma_min, sigma_max, max(2, steps))
    rng = np.random.default_rng(seed)
    coherence = []
    t0 = time.perf_counter()
    entropy: list[float] = []
    entropy_deltas: list[float] = []
    two_pi = 2.0 * math.pi

    for sigma in sigmas:
        phases = (math.pi + rng.normal(0.0, sigma, size=rotations)) % two_pi
        value = abs(np.mean(np.exp(1j * phases)))
        coherence.append(float(value))

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
    latency_ms = (time.perf_counter() - t0) * 1000.0
    discrete_steps = [
        idx
        for idx, delta in enumerate(entropy_deltas)
        if abs(delta) >= entropy_threshold and idx > 0
    ]

    result = {
        "sigma": sigmas.tolist(),
        "coherence": coherence,
        "entropy": entropy,
        "entropy_delta": entropy_deltas,
        "entropy_steps": discrete_steps,
        "threshold_index": _first_below(coherence, 0.5),
        "sigma_min": sigma_min,
        "entropy_threshold": entropy_threshold,
        "seed": seed,
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
