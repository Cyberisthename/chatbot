"""Material-time drift experiment for π-phase memories."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run


def run_pi_drift(
    *,
    rate: float,
    steps: int,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    phi = math.pi
    coherence = []
    complex_sum = 0.0 + 0.0j
    t0 = time.perf_counter()
    for step in range(max(1, steps)):
        phi += rate
        phi += rng.normal(0.0, 5e-13)
        complex_sum += complex(math.cos(phi), math.sin(phi))
        coherence.append(abs(complex_sum) / (step + 1))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "rate": rate,
        "steps": steps,
        "seed": seed,
        "coherence_final": float(coherence[-1]),
        "coherence_half_life": _half_life(coherence),
    }
    artifact_path = artifact_path or "artifacts/pi_drift.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    adapter_id = adapter_id or f"pi.drift.{seed}"
    create_adapter(adapter_id, data=result, meta={"experiment": "pi_drift"})
    log_quantum_run(
        "pi.drift",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"coherence": result["coherence_final"], "entropy": None, "S": None},
        delta_v=None,
    )
    return result


def _half_life(values: list[float]) -> int | None:
    if not values:
        return None
    initial = values[0]
    final = values[-1]
    if initial == final:
        return None
    threshold = initial - (initial - final) / 2.0
    for idx, value in enumerate(values):
        if value <= threshold:
            return idx
    return None
