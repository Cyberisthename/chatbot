"""Helpers to flatten medicinal docking metrics into JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .docking import adapter_metrics


def build_report(adapter_id: str, *, output: str | None = None) -> Dict[str, object]:
    metrics = adapter_metrics(adapter_id)
    report = {
        "adapter": adapter_id,
        "metrics": metrics,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report
