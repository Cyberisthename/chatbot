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
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    sigmas = np.linspace(0.0, sigma_max, max(2, steps))
    rng = np.random.default_rng(seed)
    coherence = []
    t0 = time.perf_counter()
    for sigma in sigmas:
        phases = math.pi + rng.normal(0.0, sigma, size=rotations)
        value = abs(np.mean(np.exp(1j * phases)))
        coherence.append(float(value))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "sigma": sigmas.tolist(),
        "coherence": coherence,
        "threshold_index": _first_below(coherence, 0.5),
        "seed": seed,
    }
    artifact_path = artifact_path or "artifacts/pi_noise_scan.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    adapter_id = adapter_id or f"pi.noise.{seed}"
    create_adapter(adapter_id, data=result, meta={"experiment": "pi_noise"})
    log_quantum_run(
        "pi.noise",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"coherence": coherence[-1], "entropy": None, "S": None},
        delta_v=None,
    )
    return result


def _first_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values):
        if value < threshold:
            return idx
    return None
