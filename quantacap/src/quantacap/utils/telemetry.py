"""Telemetry helpers for experiment logging."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Mapping

LOG_COLUMNS = (
    "timestamp",
    "experiment",
    "seed",
    "latency_ms",
    "entropy",
    "S_value",
    "coherence",
    "delta_v",
)


def log_quantum_run(
    experiment: str,
    *,
    seed: int | None,
    latency_ms: float | None,
    metrics: Mapping[str, float | int | None] | None,
    delta_v: float | None = None,
    path: str | os.PathLike[str] = "logs/quantum_runs.csv",
) -> Path:
    """Append a telemetry row to the shared CSV file.

    The function creates the destination directory on demand and writes a
    header row the first time it is invoked.
    """

    metrics = metrics or {}
    row = {
        "timestamp": f"{time.time():.6f}",
        "experiment": experiment,
        "seed": "" if seed is None else str(seed),
        "latency_ms": "" if latency_ms is None else f"{latency_ms:.3f}",
        "entropy": _fmt_metric(metrics.get("entropy")),
        "S_value": _fmt_metric(metrics.get("S")),
        "coherence": _fmt_metric(metrics.get("coherence")),
        "delta_v": _fmt_metric(delta_v),
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_path


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6e}"
