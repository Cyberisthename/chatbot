"""Simple ray-tracing through a Schwarzschild lens."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run

from .schwarzschild import GeodesicResult, integrate_null_geodesic


def render_lensing_map(
    *,
    resolution: int,
    impact_min: float,
    impact_max: float,
    seed: int = 424242,
    adapter_id: str | None = None,
    artifact_prefix: str = "artifacts/astro_ring",
) -> Dict[str, object]:
    grid = np.linspace(impact_min, impact_max, resolution)
    intensity = np.zeros((resolution, resolution), dtype=float)
    metrics: list[Dict[str, float]] = []
    for i, b_x in enumerate(grid):
        for j, b_y in enumerate(grid):
            b = math.hypot(b_x, b_y)
            geo = integrate_null_geodesic(b, steps=2000)
            intensity[i, j] = math.exp(-abs(geo.deflection))
            metrics.append({"b": b, "deflection": geo.deflection})
    intensity /= intensity.max() if intensity.max() else 1.0
    artifact_prefix = Path(artifact_prefix)
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    image_path = artifact_prefix.with_suffix(".npy")
    np.save(image_path, intensity)
    meta = {
        "resolution": resolution,
        "impact_min": impact_min,
        "impact_max": impact_max,
        "intensity_path": str(image_path),
    }
    with artifact_prefix.with_suffix("_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    adapter_id = adapter_id or f"astro.lens.{resolution}.{impact_min:.2f}_{impact_max:.2f}"
    create_adapter(adapter_id, data={"meta": meta}, meta={"experiment": "astro_lens"})
    log_quantum_run(
        "astro.lens",
        seed=seed,
        latency_ms=None,
        metrics={"S": None, "entropy": None, "coherence": None},
        delta_v=None,
    )
    return meta
