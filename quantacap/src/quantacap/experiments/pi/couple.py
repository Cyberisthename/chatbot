"""Coupling experiment for π-locked oscillators."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run


def run_pi_coupling(
    *,
    kappa: float,
    steps: int,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    phi = np.array([math.pi, math.pi + 1e-6], dtype=float)
    history = []
    sync_threshold = 1e-8
    sync_step = None
    t0 = time.perf_counter()
    for step in range(max(1, steps)):
        diff = phi[1] - phi[0]
        phi[0] += kappa * diff
        phi[1] -= kappa * diff
        phi += rng.normal(0.0, 1e-12, size=2)
        phi = np.mod(phi, 2 * math.pi)
        history.append(phi.copy())
        if sync_step is None and abs(diff) < sync_threshold:
            sync_step = step
    history = np.asarray(history)
    coherence = float(abs(np.mean(np.exp(1j * history), axis=0)).mean())
    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "kappa": kappa,
        "steps": steps,
        "seed": seed,
        "sync_step": sync_step,
        "coherence": coherence,
        "phase_mean": history.mean(axis=0).tolist(),
    }

    artifact_path = artifact_path or "artifacts/pi_couple.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    adapter_id = adapter_id or f"pi.couple.{seed}"
    create_adapter(adapter_id, data=result, meta={"experiment": "pi_couple"})
    log_quantum_run(
        "pi.couple",
        seed=seed,
        latency_ms=latency_ms,
        metrics={"coherence": coherence, "entropy": None, "S": None},
        delta_v=None,
    )
    return result
