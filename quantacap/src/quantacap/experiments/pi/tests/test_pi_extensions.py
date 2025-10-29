from __future__ import annotations


import pytest

from quantacap.experiments.pi.couple import run_pi_coupling
from quantacap.experiments.pi.drift import run_pi_drift
from quantacap.experiments.pi.noise import run_pi_noise_scan, run_pi_entropy_collapse
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
    result = run_pi_noise_scan(
        sigma_max=1e-9,
        sigma_min=1e-12,
        steps=5,
        rotations=128,
        entropy_threshold=0.0,
        seed=5,
    )
    assert len(result["sigma"]) == len(result["coherence"])
    assert len(result["entropy"]) == len(result["sigma"])
    assert result["sigma"][0] == pytest.approx(1e-12)


def test_pi_entropy_control() -> None:
    result = run_pi_entropy_control(steps=100, seed=7)
    assert result["entropy_final"] >= 0


def test_pi_entropy_collapse_steps() -> None:
    result = run_pi_entropy_collapse(
        kappa=0.02,
        sigma_min=1e-9,
        sigma_max=1e-7,
        stages=6,
        stage_length=32,
        entropy_threshold=0.0,
        seed=11,
    )
    assert len(result["sigma"]) == 6
    assert len(result["entropy"]) == 6
    if result["entropy_steps"]:
        assert all(idx >= 0 for idx in result["entropy_steps"])
