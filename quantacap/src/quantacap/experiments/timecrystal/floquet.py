"""Discrete time-crystal style Floquet simulation (toy model)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from quantacap.utils.optional_import import optional_import


def _np():
    return optional_import("numpy", pip_name="numpy", purpose="time-crystal simulation")


@dataclass
class TimeCrystalResult:
    autocorrelation: List[float]
    magnetisation: List[float]
    frequencies: List[float]
    spectrum: List[float]
    subharmonic_peak_hz: float
    subharmonic_strength: float
    detected: bool
    parameters: Dict[str, float]


def run_time_crystal(
    *,
    N: int = 10,
    steps: int = 500,
    disorder: float = 0.08,
    jitter: float = 0.05,
    seed: int = 424242,
) -> TimeCrystalResult:
    """Simulate a synthetic Floquet map exhibiting period-doubling."""

    if N <= 0 or steps <= 0:
        raise ValueError("N and steps must be positive")

    np = _np()
    rng = np.random.default_rng(seed)

    spins = np.ones(N, dtype=float)
    reference = spins.copy()
    magnetisation = []
    autocorr = []

    for t in range(steps):
        global_flip = -1.0 if t % 2 == 0 else 1.0
        jit = 1.0 + rng.normal(0.0, jitter)
        spins *= global_flip * jit
        # local disorder kicks
        defects = rng.random(N) < disorder
        spins[defects] *= -1.0
        spins = np.sign(spins)
        magnetisation.append(float(spins.mean()))
        autocorr.append(float(np.mean(reference * spins)))

    autocorr = np.asarray(autocorr)
    spectrum = np.abs(np.fft.rfft(autocorr))
    freqs = np.fft.rfftfreq(len(autocorr), d=1.0)
    if spectrum.size > 0:
        spectrum[0] = 0.0
    target_idx = int(np.argmin(np.abs(freqs - 0.5))) if freqs.size else 0
    subharmonic_strength = float(spectrum[target_idx]) if spectrum.size else 0.0
    peak_idx = int(np.argmax(spectrum)) if spectrum.size else 0
    subharmonic_peak = float(freqs[peak_idx]) if freqs.size else 0.0
    detected = spectrum.size > 0 and abs(subharmonic_peak - 0.5) < 0.08

    return TimeCrystalResult(
        autocorrelation=autocorr.tolist(),
        magnetisation=magnetisation,
        frequencies=freqs.tolist(),
        spectrum=spectrum.tolist(),
        subharmonic_peak_hz=subharmonic_peak,
        subharmonic_strength=subharmonic_strength,
        detected=bool(detected),
        parameters={
            "N": float(N),
            "steps": float(steps),
            "disorder": float(disorder),
            "jitter": float(jitter),
            "seed": float(seed),
        },
    )
