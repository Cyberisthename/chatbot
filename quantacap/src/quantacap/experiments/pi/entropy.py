"""Entropy minimisation control loop for π-phase rotations."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run


def run_pi_entropy_control(
    *,
    steps: int,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    phases = math.pi + rng.normal(0.0, 1e-9, size=steps)
    control = 0.0
    entropy_trace = []
    t0 = time.perf_counter()
    for idx in range(steps):
        control += -0.1 * (phases[idx] - math.pi)
        adjusted = phases[idx] + control
        entropy_trace.append(_phase_entropy(adjusted))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "steps": steps,
        "seed": seed,
        "entropy_final": float(entropy_trace[-1]),
        "entropy_mean": float(np.mean(entropy_trace)),
    }
    artifact_path = artifact_path or "artifacts/pi_entropy.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    adapter_id = adapter_id or f"pi.entropy.{seed}"
    create_adapter(adapter_id, data=result, meta={"experiment": "pi_entropy"})
    log_quantum_run(
        "pi.entropy",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"entropy": result["entropy_final"], "coherence": None, "S": None},
        delta_v=None,
    )
    return result


def _phase_entropy(value: float) -> float:
    centre = abs(((value + math.pi) % (2 * math.pi)) - math.pi)
    return float(centre**2)
