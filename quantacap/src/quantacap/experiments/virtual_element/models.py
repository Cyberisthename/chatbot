"""Model heuristics for virtual superheavy nuclei (for research only).

These routines implement semi-empirical mass formulas combined with a
simple shell-correction ansatz to produce *theoretical* stability
estimates.  They are not predictions of synthesised matter and are
intended for safe, offline numerical exploration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

from quantacap.utils.optional_import import optional_import


def _np():
    return optional_import(
        "numpy", pip_name="numpy", purpose="virtual element calculations"
    )


@dataclass(frozen=True)
class ModelParams:
    """Container for tunable coefficients used by the models."""

    weizsacker: Mapping[str, float]
    shell: Mapping[str, float]
    stability: Mapping[str, float]


DEFAULT_WEIZSACKER_PARAMS: Dict[str, float] = {
    "a_v": 15.75,  # volume term (MeV)
    "a_s": 17.8,  # surface term (MeV)
    "a_c": 0.711,  # Coulomb term (MeV)
    "a_a": 23.7,  # asymmetry term (MeV)
    "a_p": 12.0,  # pairing term (MeV)
}

DEFAULT_SHELL_PARAMS: Dict[str, float] = {
    "strength": 5.0,  # MeV contribution near magic numbers
    "width_Z": 6.0,
    "width_N": 8.0,
    "magic_Z1": 114.0,
    "magic_Z2": 120.0,
    "magic_Z3": 126.0,
    "magic_N1": 184.0,
    "magic_N2": 196.0,
}

DEFAULT_STABILITY_PARAMS: Dict[str, float] = {
    "sf_scale": 0.25,
    "sf_Z_ref": 114.0,
    "sf_N_ref": 184.0,
    "sf_width_Z": 18.0,
    "sf_width_N": 24.0,
    "qalpha_norm": 12.0,
    "be_norm": 8.5,
}

_ALPHA_BE: float | None = None


def _resolve_params(params: Mapping[str, Mapping[str, float]] | None) -> ModelParams:
    weiz = dict(DEFAULT_WEIZSACKER_PARAMS)
    shell = dict(DEFAULT_SHELL_PARAMS)
    stab = dict(DEFAULT_STABILITY_PARAMS)
    if params:
        weiz.update(params.get("weizsacker", {}))
        shell.update(params.get("shell", {}))
        stab.update(params.get("stability", {}))
    return ModelParams(weiz, shell, stab)


def _pairing_term(A: int, Z: int, a_p: float) -> float:
    if A % 2 == 1:
        return 0.0
    N = A - Z
    if Z % 2 == 0 and N % 2 == 0:
        return +a_p / math.sqrt(A)
    if Z % 2 == 1 and N % 2 == 1:
        return -a_p / math.sqrt(A)
    return 0.0


def weizsacker_binding_energy(
    Z: int, A: int, params: Mapping[str, Mapping[str, float]] | None = None
) -> float:
    """Return the liquid-drop binding energy estimate in MeV."""

    if A <= 0 or Z <= 0 or Z > A:
        raise ValueError("invalid nucleus specification")
    coeffs = _resolve_params(params).weizsacker
    np = _np()
    av = float(coeffs["a_v"])
    a_s = float(coeffs["a_s"])
    a_c = float(coeffs["a_c"])
    a_a = float(coeffs["a_a"])
    a_p = float(coeffs["a_p"])

    A_third = A ** (1.0 / 3.0)
    surface = A ** (2.0 / 3.0)
    asym = (A - 2 * Z) ** 2

    term_volume = av * A
    term_surface = a_s * surface
    term_coulomb = a_c * Z * (Z - 1) / A_third
    term_asym = a_a * asym / A
    term_pair = _pairing_term(A, Z, a_p)

    binding = term_volume - term_surface - term_coulomb - term_asym + term_pair
    return float(binding)


def shell_correction_heuristic(
    Z: int, A: int, params: Mapping[str, Mapping[str, float]] | None = None
) -> float:
    """Return a phenomenological shell correction in MeV."""

    cfg = _resolve_params(params).shell
    np = _np()
    N = A - Z
    strength = float(cfg["strength"])
    width_Z = float(cfg["width_Z"])
    width_N = float(cfg["width_N"])

    def gaussian(x: float, mu: float, width: float) -> float:
        return float(np.exp(-((x - mu) ** 2) / (2.0 * width**2)))

    bonus_Z = 0.0
    for key in ("magic_Z1", "magic_Z2", "magic_Z3"):
        bonus_Z += gaussian(Z, float(cfg[key]), width_Z)

    bonus_N = 0.0
    for key in ("magic_N1", "magic_N2"):
        bonus_N += gaussian(N, float(cfg[key]), width_N)

    return strength * (0.6 * bonus_Z + 0.4 * bonus_N)


def _alpha_binding_energy() -> float:
    global _ALPHA_BE
    if _ALPHA_BE is None:
        _ALPHA_BE = weizsacker_binding_energy(2, 4)
    return _ALPHA_BE


def combined_binding_energy(
    Z: int,
    A: int,
    params: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, float]:
    """Return binding metrics and a synthetic stability score."""

    np = _np()
    resolved = _resolve_params(params)
    binding_ld = weizsacker_binding_energy(Z, A, params)
    delta_shell = shell_correction_heuristic(Z, A, params)
    total_binding = binding_ld + delta_shell
    binding_per_A = total_binding / A

    if Z > 2 and A > 4:
        daughter = combined_binding_energy(Z - 2, A - 4, params)
        q_alpha = total_binding - daughter["binding_energy_MeV"] - _alpha_binding_energy()
    else:
        q_alpha = float("nan")

    # Spontaneous fission vulnerability heuristic (0..1 roughly)
    stab_cfg = resolved.stability
    N = A - Z
    sf_Z = float(stab_cfg["sf_scale"]) * np.exp(
        -((Z - stab_cfg["sf_Z_ref"]) ** 2) / (2.0 * stab_cfg["sf_width_Z"] ** 2)
    )
    sf_N = float(stab_cfg["sf_scale"]) * np.exp(
        -((N - stab_cfg["sf_N_ref"]) ** 2) / (2.0 * stab_cfg["sf_width_N"] ** 2)
    )
    sf_vulnerability = float(np.clip(1.0 - (sf_Z + sf_N), 0.0, 1.0))

    qalpha_norm = float(stab_cfg["qalpha_norm"])
    be_norm = float(stab_cfg["be_norm"])
    q_penalty = 1.0 / (1.0 + abs(q_alpha) / qalpha_norm) if np.isfinite(q_alpha) else 0.5
    shell_bonus = 0.5 + delta_shell / 8.0
    sf_bonus = 1.0 - sf_vulnerability
    stability_score = float(
        0.55 * (binding_per_A / be_norm)
        + 0.25 * q_penalty
        + 0.15 * shell_bonus
        + 0.05 * sf_bonus
    )

    return {
        "binding_energy_MeV": float(total_binding),
        "binding_energy_per_A_MeV": float(binding_per_A),
        "binding_energy_LD_MeV": float(binding_ld),
        "delta_shell_MeV": float(delta_shell),
        "Q_alpha_MeV": float(q_alpha),
        "sf_vulnerability": float(sf_vulnerability),
        "stability_score": stability_score,
    }
