"""Generate 3D scalar fields representing synthetic computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from quantacap.core.adapter_store import load_adapter


@dataclass
class FieldFrame:
    step: int
    field: np.ndarray
    metadata: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "metadata": self.metadata,
            "field": self.field.tolist(),
        }


def build_field_series(
    *,
    source: str,
    field: str,
    grid: Tuple[int, int, int],
    steps: int,
    seed: int = 424242,
) -> Iterable[FieldFrame]:
    rng = np.random.default_rng(seed)
    base = _initial_volume(source, grid)
    frames: list[FieldFrame] = []
    volume = base.copy()
    for step in range(max(1, steps)):
        fluctuation = rng.normal(0.0, 0.05, size=grid)
        volume = 0.9 * volume + 0.1 * fluctuation
        if field == "amplitude":
            mapped = np.abs(volume)
        elif field == "phase":
            mapped = np.mod(np.angle(volume + 1j * 0.0), 2 * np.pi)
        elif field == "entropy":
            probs = np.abs(volume)**2
            total = probs.sum()
            probs = probs / total if total else probs
            mapped = -(probs * np.log(probs + 1e-12))
        else:
            raise ValueError(f"Unsupported field '{field}'")
        frames.append(
            FieldFrame(
                step=step,
                field=mapped,
                metadata={
                    "min": float(mapped.min()),
                    "max": float(mapped.max()),
                    "mean": float(mapped.mean()),
                },
            )
        )
    return frames


def _initial_volume(source: str, grid: Tuple[int, int, int]) -> np.ndarray:
    if source.startswith("adapter:"):
        adapter_id = source.split(":", 1)[1]
        try:
            record = load_adapter(adapter_id)
        except FileNotFoundError:
            return np.zeros(grid)
        probs = _extract_probs(record)
        if probs.size:
            return probs.reshape(grid) if probs.size == np.prod(grid) else np.resize(probs, grid)
    rng = np.random.default_rng(424242)
    return rng.random(grid)


def _extract_probs(record: Dict[str, object]) -> np.ndarray:
    data = record.get("data", {}) if isinstance(record, dict) else {}
    state = data.get("state") if isinstance(data, dict) else None
    if isinstance(state, dict) and "probs" in state:
        return np.asarray(state["probs"], dtype=float)
    if "counts" in data:
        values = np.array(list(data["counts"].values()), dtype=float)
        total = values.sum()
        if total:
            return values / total
    return np.array([], dtype=float)
