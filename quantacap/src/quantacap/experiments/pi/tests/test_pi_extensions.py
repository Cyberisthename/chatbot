from __future__ import annotations


import pytest

from quantacap.experiments.pi.couple import run_pi_coupling
from quantacap.experiments.pi.drift import run_pi_drift
from quantacap.experiments.pi.noise import run_pi_noise_scan
from quantacap.experiments.pi.entropy import run_pi_entropy_control


@pytest.mark.parametrize("kappa", [0.01, 0.05])
def test_pi_coupling_sync(kappa: float) -> None:
    result = run_pi_coupling(kappa=kappa, steps=50, seed=123)
    assert result["sync_step"] is not None
    assert 0 <= result["sync_step"] < 50


def test_pi_drift_half_life() -> None:
    result = run_pi_drift(rate=1e-6, steps=100, seed=12)
    assert result["coherence_half_life"] is None or result["coherence_half_life"] >= 0


def test_pi_noise_scan_threshold() -> None:
    result = run_pi_noise_scan(sigma_max=1e-9, steps=5, rotations=128, seed=5)
    assert len(result["sigma"]) == len(result["coherence"])


def test_pi_entropy_control() -> None:
    result = run_pi_entropy_control(steps=100, seed=7)
    assert result["entropy_final"] >= 0
