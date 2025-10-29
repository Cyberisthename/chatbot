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


def integrate_null_geodesic(b: float, *, steps: int = 5000, phi_max: float = math.pi) -> GeodesicResult:
    u = 0.0
    du = -1.0 / b
    h = phi_max / max(1, steps)
    phi = [0.0]
    r_vals = [math.inf]
    captured = False
    closest = math.inf

    for _ in range(steps):
        def f(phi_val: float, state: np.ndarray) -> np.ndarray:
            u_val, du_val = state
            d2u = 3 * M * u_val**2 - u_val
            return np.array([du_val, d2u])

        state = np.array([u, du])
        k1 = f(0, state)
        k2 = f(0, state + 0.5 * h * k1)
        k3 = f(0, state + 0.5 * h * k2)
        k4 = f(0, state + h * k3)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u, du = state
        current_phi = phi[-1] + h
        phi.append(current_phi)
        if u <= 0:
            r = math.inf
        else:
            r = 1.0 / u
            closest = min(closest, r)
            if r <= R_S * 1.01:
                captured = True
                break
        r_vals.append(r)
        if r > 50 and len(phi) > steps // 2:
            break

    deflection = current_phi * 2 - math.pi
    return GeodesicResult(
        impact_parameter=b,
        deflection=float(deflection),
        closest_approach=float(closest),
        captured=captured,
        phi=[float(val) for val in phi],
        r=[float(val) for val in r_vals],
    )
