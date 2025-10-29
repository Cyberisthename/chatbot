"""Simple ray-tracing through a Schwarzschild lens."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.telemetry import log_quantum_run
from quantacap.utils.optional_import import optional_import

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
    png_path = _maybe_write_png(artifact_prefix.with_suffix(".png"), intensity)
    meta = {
        "resolution": resolution,
        "impact_min": impact_min,
        "impact_max": impact_max,
        "intensity_path": str(image_path),
        "png_path": str(png_path) if png_path else None,
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


def run_lens_atom_equivalence(
    *,
    resolution: int,
    impact_min: float,
    impact_max: float,
    atom_artifact: str,
    adapter_id: str | None = None,
) -> Dict[str, object]:
    """Render a micro-scale lens and compare it with an atom density map."""

    meta = render_lensing_map(
        resolution=resolution,
        impact_min=impact_min,
        impact_max=impact_max,
        adapter_id=adapter_id,
        artifact_prefix=f"artifacts/astro_equivalence_{resolution}",
    )

    intensity = np.load(meta["intensity_path"])
    atom = _load_atom_density(atom_artifact)
    resized_atom = np.resize(atom, intensity.shape)

    lens_norm = intensity / intensity.sum() if intensity.sum() else intensity
    atom_norm = resized_atom / resized_atom.sum() if resized_atom.sum() else resized_atom

    correlation = _safe_correlation(lens_norm, atom_norm)
    similarity = float((np.minimum(lens_norm, atom_norm)).sum())

    report = {
        "correlation": correlation,
        "overlap": similarity,
        "lens_meta": meta,
        "atom_artifact": atom_artifact,
    }

    out_path = Path("artifacts/astro_equivalence_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    create_adapter(
        adapter_id or f"astro.equivalence.{resolution}",
        data=report,
        meta={"experiment": "astro_equivalence"},
    )
    log_quantum_run(
        "astro.equivalence",
        seed=None,
        latency_ms=None,
        metrics={"S": None, "entropy": None, "coherence": report["correlation"]},
        delta_v=None,
    )
    return report


def _maybe_write_png(path: Path, intensity: np.ndarray) -> Path | None:
    try:
        mpl = optional_import(
            "matplotlib", pip_name="matplotlib", purpose="export Einstein ring maps"
        )
    except RuntimeError:
        return None
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(4, 4))
    plt.imshow(intensity, cmap="inferno", origin="lower")
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return path


def _load_atom_density(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    density = data.get("density")
    if density is None and "trace" in data:
        density = data["trace"].get("density")
    array = np.asarray(density, dtype=float)
    return array.reshape(array.shape) if array.size else array


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return corr
