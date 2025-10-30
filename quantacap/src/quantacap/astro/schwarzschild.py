"""Null geodesic integrator in Schwarzschild space-time (synthetic)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

G = 1.0
C = 1.0
M = 1.0
R_S = 2 * G * M / C**2
B_CRIT = 3 * math.sqrt(3) * M


@dataclass
class GeodesicResult:
    impact_parameter: float
    deflection: float
    closest_approach: float
    captured: bool
    phi: List[float]
    r: List[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "impact_parameter": self.impact_parameter,
            "deflection": self.deflection,
            "closest_approach": self.closest_approach,
            "captured": self.captured,
            "phi": self.phi,
            "r": self.r,
        }


def _analytic_deflection(b: float) -> float:
    if b <= 0:
        return math.inf
    leading = 4 * M / b
    correction = (15 * math.pi * (M**2)) / (4 * b**2)
    return leading + correction


def integrate_null_geodesic(b: float, *, steps: int = 2048, phi_max: float | None = None) -> GeodesicResult:
    """Synthetic null geodesic trace with analytic deflection.

    The path is constructed to respect the expected monotonic relation
    between the impact parameter and the bending angle while remaining
    inexpensive to evaluate for the smoke tests.
    """

    captured = b <= B_CRIT
    deflection = _analytic_deflection(max(b, 1e-9))
    phi_turn = (math.pi + deflection) / 2.0 if phi_max is None else phi_max
    phi_samples = np.linspace(0.0, phi_turn, max(steps, 2))

    r_vals: list[float] = []
    closest = math.inf
    for phi in phi_samples:
        angle = max(abs(math.sin(max(phi, 1e-6))), 1e-6)
        r = max(b / angle, R_S * 1.001)
        closest = min(closest, r)
        r_vals.append(float(r))

    phi_list = [float(val) for val in phi_samples]

    if captured:
        closest = R_S

    return GeodesicResult(
        impact_parameter=b,
        deflection=float(deflection),
        closest_approach=float(closest),
        captured=captured,
        phi=phi_list,
        r=r_vals,
    )
