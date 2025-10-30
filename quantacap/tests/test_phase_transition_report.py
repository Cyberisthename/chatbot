"""Smoke test for the phase transition report script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_numpy_spec = importlib.util.find_spec("numpy")
pytestmark = pytest.mark.skipif(_numpy_spec is None, reason="NumPy not installed")

if _numpy_spec is not None:  # pragma: no cover - guarded import
    from quantacap.scripts.phase_transition_report import main as phase_main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.skipif(_numpy_spec is None, reason="NumPy not installed")
def test_phase_transition_summary_schema(tmp_path: Path) -> None:
    artifacts = tmp_path
    # Noise sweeps
    noise_payload = {
        "seed": 424242,
        "sigma": [1e-12, 1e-10, 1e-8],
        "entropy": [0.1, 0.15, 0.4],
        "entropy_steps": [1, 2],
        "entropy_plateaus": [
            {"start": 0, "end": 1, "mean_entropy": 0.125},
            {"start": 2, "end": 2, "mean_entropy": 0.4},
        ],
    }
    _write_json(artifacts / "pi_noise_up.json", noise_payload)
    noise_down = dict(noise_payload)
    noise_down["entropy_steps"] = [0]
    _write_json(artifacts / "pi_noise_down.json", noise_down)

    # Coupling ladder
    for kappa, sync in [(0.01, 400), (0.05, 120), (0.08, 30)]:
        _write_json(
            artifacts / f"pi_couple_k_{kappa}.json",
            {"kappa": kappa, "sync_step": sync},
        )

    # Drift sweeps
    for rate, half_life in [(1e-7, 80000), (3e-7, 78000), (1e-6, 77000)]:
        _write_json(
            artifacts / f"pi_drift_r_{rate}.json",
            {"rate": rate, "coherence_half_life": half_life},
        )

    summary = phase_main(artifacts_dir=artifacts, output_prefix=artifacts / "phase", enable_plots=False)

    summary_path = artifacts / "phase_transition_summary.json"
    bundle_path = artifacts / "phase_transition_bundle.zip"

    assert summary_path.is_file()
    assert bundle_path.is_file()

    assert "hysteresis" in summary and "frontier" in summary and "drift" in summary
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["hysteresis"]["steps_up"]
    assert loaded["frontier"]["sync_curve"]
    assert loaded["drift"]["points"]
