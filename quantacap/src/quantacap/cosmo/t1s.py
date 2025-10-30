r"""Estimate the early-universe energy budget at :math:`t=1\,\text{s}`.

This module assumes a radiation dominated Friedman–Robertson–Walker universe
where the temperature is approximately :math:`T\approx 1` MeV at one second
after the Big Bang.  Degrees of freedom are parameterised by ``g_star`` and the
calculation follows the textbook radiation energy density expression with
fundamental constants in SI units.
"""

from __future__ import annotations

import math
from typing import Dict

PI = math.pi
K_B = 1.380_649e-23  # J K^-1
H_BAR = 1.054_571_817e-34  # J s
C_LIGHT = 2.997_924_58e8  # m s^-1
MEV_TO_J = 1.602_176_634e-13  # J
MEV_TO_K = 1.160_45e10  # K per MeV
RADIATION_CONSTANT = 7.5657e-16  # J m^-3 K^-4 for photons (g=2)
T0_CMB_K = 2.7255
G_STAR_TODAY = 3.36  # photons + relativistic neutrinos today
ERG_PER_J = 1.0e7


def _radiation_density_exact(g_star: float, temperature_K: float, *, c: float) -> float:
    """Return radiation energy density using the exact relativistic expression."""

    kbt = K_B * temperature_K
    return (PI**2 / 30.0) * g_star * (kbt**4) / ((H_BAR**3) * (c**3))


def _radiation_density_parametric(g_star: float, temperature_K: float) -> float:
    """Approximate density using the photon radiation constant scaled by ``g_star``."""

    return RADIATION_CONSTANT * (g_star / 2.0) * (temperature_K**4)


def _modern_scaled_density(g_star: float, temperature_K: float, *, c: float) -> float:
    """Scale today's radiation density back to the requested temperature."""

    rho_today = _radiation_density_exact(G_STAR_TODAY, T0_CMB_K, c=c)
    scale = (temperature_K / T0_CMB_K) ** 4
    return rho_today * scale * (g_star / G_STAR_TODAY)


def energy_at_1s(
    t: float = 1.0,
    g_star: float = 10.75,
    T_MeV: float = 1.0,
    c: float = C_LIGHT,
) -> Dict[str, float | str]:
    """Estimate the energy inside the causal horizon at ``t`` seconds."""

    temperature_K = T_MeV * MEV_TO_K
    rho_exact = _radiation_density_exact(g_star, temperature_K, c=c)
    rho_param = _radiation_density_parametric(g_star, temperature_K)
    rho_modern = _modern_scaled_density(g_star, temperature_K, c=c)

    horizon_radius = c * t
    horizon_volume = (4.0 / 3.0) * PI * (horizon_radius**3)

    energy_total = rho_exact * horizon_volume

    notes = (
        "Radiation dominated FRW model with T≈1 MeV at t≈1 s. "
        "Degrees of freedom g*=10.75 (photons, e±, ν). "
        "Temperature conversion uses 1 MeV ≈ 1.16045e10 K."
    )

    return {
        "time_s": float(t),
        "g_star": float(g_star),
        "T_MeV": float(T_MeV),
        "T_K": float(temperature_K),
        "rho_J_m3": float(rho_exact),
        "rho_parametric_J_m3": float(rho_param),
        "rho_modern_scaled_J_m3": float(rho_modern),
        "rho_ratio_vs_modern_scaled": float(rho_exact / rho_modern if rho_modern else math.nan),
        "horizon_radius_m": float(horizon_radius),
        "horizon_volume_m3": float(horizon_volume),
        "E_total_J": float(energy_total),
        "E_total_erg": float(energy_total * ERG_PER_J),
        "notes": notes,
    }


__all__ = ["energy_at_1s"]
