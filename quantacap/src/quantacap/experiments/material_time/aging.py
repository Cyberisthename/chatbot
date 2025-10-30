"""Synthetic material-time aging simulation (toy trap model)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from quantacap.utils.optional_import import optional_import


def _np():
    return optional_import("numpy", pip_name="numpy", purpose="material-time simulation")


@dataclass
class MaterialTimeResult:
    real_time: List[int]
    material_time: List[int]
    entropy_proxy: List[float]
    reversal_attempts: int
    reversal_success: int
    parameters: Dict[str, float]


def simulate_material_time(
    *,
    traps: int = 64,
    steps: int = 5000,
    rate: float = 1e-3,
    temperature: float = 0.3,
    seed: int = 424242,
    reversal_period: int = 250,
) -> MaterialTimeResult:
    """Run a kinetically constrained trap model with a material-time ticker."""

    if traps <= 0 or steps <= 0:
        raise ValueError("traps and steps must be positive")

    np = _np()
    rng = np.random.default_rng(seed)

    barriers = rng.exponential(scale=1.0, size=traps)
    visits = np.zeros(traps, dtype=int)
    current = rng.integers(0, traps)
    material_clock = 0
    real_time: List[int] = []
    material_series: List[int] = []
    entropy_series: List[float] = []
    reversal_attempts = 0
    reversal_success = 0
    last_event = current

    for t in range(steps):
        barrier = barriers[current]
        hop_prob = min(1.0, rate * np.exp(-barrier / max(temperature, 1e-6)))
        hop = rng.random() < hop_prob
        if hop:
            material_clock += 1
            visits[current] += 1
            last_event = current
            current = rng.integers(0, traps)
        else:
            visits[current] += 1

        total_visits = visits.sum()
        if total_visits > 0:
            probs = visits / total_visits
            entropy = float(-np.sum(np.where(probs > 0, probs * np.log(probs), 0.0)))
        else:
            entropy = 0.0

        if reversal_period and (t + 1) % reversal_period == 0:
            reversal_attempts += 1
            if current == last_event:
                reversal_success += 1
            else:
                # emulate irreversible rearrangement by randomising history
                visits[:] = np.maximum(visits - 1, 0)

        real_time.append(t)
        material_series.append(material_clock)
        entropy_series.append(entropy)

    return MaterialTimeResult(
        real_time=real_time,
        material_time=material_series,
        entropy_proxy=entropy_series,
        reversal_attempts=reversal_attempts,
        reversal_success=reversal_success,
        parameters={
            "traps": float(traps),
            "steps": float(steps),
            "rate": float(rate),
            "temperature": float(temperature),
            "seed": float(seed),
        },
    )
