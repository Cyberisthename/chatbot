"""Probabilistic phase rotations that encode approximations of pi."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from quantacap.core.adapter_store import create_adapter


@dataclass(frozen=True)
class PiPhaseSummary:
    rotations: int
    precision: float
    seed: int
    alignment_ratio: float
    phase_lock_ratio: float
    stability_score: float
    mean_rotation: float
    rotation_std: float
    tail_std: float
    phase_rms_error: float
    coherence: float
    stabilized: bool

    def as_dict(self) -> Dict[str, float | int | bool]:
        return {
            "rotations": self.rotations,
            "precision": self.precision,
            "seed": self.seed,
            "alignment_ratio": self.alignment_ratio,
            "phase_lock_ratio": self.phase_lock_ratio,
            "stability_score": self.stability_score,
            "mean_rotation": self.mean_rotation,
            "rotation_std": self.rotation_std,
            "tail_std": self.tail_std,
            "phase_rms_error": self.phase_rms_error,
            "coherence": self.coherence,
            "stabilized": self.stabilized,
        }


def _phase_error(value: float, base: float) -> float:
    k = round(value / base)
    return abs(value - k * base)


def _circular_difference(phase: float, target: float) -> float:
    two_pi = 2.0 * math.pi
    return ((target - phase + math.pi) % two_pi) - math.pi


def _downsample(series: np.ndarray, steps: int) -> List[int]:
    if steps <= 0:
        return []
    stride = max(1, len(series) // steps)
    indices = list(range(stride - 1, len(series), stride))
    if indices and indices[-1] != len(series) - 1:
        indices.append(len(series) - 1)
    return indices


def run_pi_phase(
    *,
    rotations: int,
    precision: float,
    seed: int = 424242,
    samples: int = 256,
    adapter_id: str | None = None,
    artifact_path: str | None = None,
) -> Dict[str, object]:
    if rotations <= 0:
        raise ValueError("rotations must be positive")
    if precision <= 0.0:
        raise ValueError("precision must be positive")

    rng = np.random.default_rng(seed)
    base = math.pi
    two_pi = 2.0 * math.pi

    rotation_values = np.empty(rotations, dtype=np.float64)
    phases = np.empty(rotations, dtype=np.float64)
    phase_errors = np.empty(rotations, dtype=np.float64)

    phase = 0.0
    alignments = 0

    damping = 0.05
    jitter_floor = precision * 1e-3

    for idx in range(rotations):
        harmonic = 0.25 * math.sin((idx + 1) * 0.013)
        jitter_scale = max(jitter_floor, precision * (1.0 + harmonic))
        jitter = float(rng.normal(0.0, jitter_scale))
        rotation = base + jitter
        phase = (phase + rotation) % two_pi

        target = base if idx % 2 == 0 else 0.0
        correction = damping * _circular_difference(phase, target)
        phase = (phase + correction) % two_pi

        rotation_values[idx] = rotation
        phases[idx] = phase

        deviation = abs(rotation - base)
        phase_err = min(_phase_error(phase, base), _phase_error((phase + base) % two_pi, base))
        phase_errors[idx] = phase_err

        if deviation <= precision:
            alignments += 1

    rotation_mean = float(rotation_values.mean())
    rotation_std = float(rotation_values.std())
    tail_window = min(rotations, 1024)
    tail_std = float(rotation_values[-tail_window:].std())

    alignment_ratio = alignments / rotations
    phase_lock_ratio = float(np.mean(phase_errors <= precision * 10.0))
    phase_rms_error = float(np.sqrt(np.mean(phase_errors**2)))

    coherence = float(np.abs(np.exp(1j * phases).mean()))
    stability_score = float(coherence * math.exp(-tail_std / base))
    stabilized = bool(phase_rms_error <= precision * 20.0 or stability_score > 0.9)

    hist_counts, hist_edges = np.histogram(phases, bins=36, range=(0.0, two_pi), density=True)
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    sample_indices = _downsample(phases, samples)
    sample_records: List[Dict[str, float | int]] = []
    for idx in sample_indices:
        sample_records.append(
            {
                "step": int(idx + 1),
                "phase": float(phases[idx]),
                "rotation": float(rotation_values[idx]),
                "phase_error": float(phase_errors[idx]),
            }
        )

    summary = PiPhaseSummary(
        rotations=rotations,
        precision=precision,
        seed=seed,
        alignment_ratio=float(alignment_ratio),
        phase_lock_ratio=phase_lock_ratio,
        stability_score=stability_score,
        mean_rotation=rotation_mean,
        rotation_std=rotation_std,
        tail_std=tail_std,
        phase_rms_error=phase_rms_error,
        coherence=coherence,
        stabilized=stabilized,
    )

    payload: Dict[str, object] = {
        "summary": summary.as_dict(),
        "histogram": {
            "bins": hist_centers.tolist(),
            "density": hist_counts.tolist(),
        },
        "samples": sample_records,
    }

    os.makedirs("artifacts", exist_ok=True)
    artifact_name = artifact_path or os.path.join(
        "artifacts", f"pi_phase_{adapter_id or 'default'}.json"
    )
    with open(artifact_name, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if adapter_id:
        trimmed_samples = sample_records[:64]
        create_adapter(
            adapter_id,
            data={
                "experiment": "pi-phase",
                "summary": summary.as_dict(),
                "stability_score": summary.stability_score,
                "alignment_ratio": summary.alignment_ratio,
                "phase_lock_ratio": summary.phase_lock_ratio,
                "coherence": summary.coherence,
                "samples": trimmed_samples,
            },
            meta={
                "rotations": rotations,
                "precision": precision,
                "seed": seed,
            },
        )

    payload["artifact"] = artifact_name
    return payload


__all__ = ["run_pi_phase", "PiPhaseSummary"]
