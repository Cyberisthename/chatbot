import math
import os

import pytest

pytest.importorskip("numpy")

from quantacap.experiments.pi_phase import run_pi_phase


def test_pi_phase_alignment(tmp_path):
    artifact = tmp_path / "pi_phase.json"
    result = run_pi_phase(
        rotations=4096,
        precision=1e-9,
        seed=424242,
        samples=16,
        adapter_id=None,
        artifact_path=str(artifact),
    )
    summary = result["summary"]

    assert math.isclose(summary["mean_rotation"], math.pi, rel_tol=1e-3)
    assert summary["alignment_ratio"] > 0.5
    assert summary["phase_lock_ratio"] > 0.5
    assert summary["stability_score"] > 0.5
    assert os.path.isfile(result["artifact"])


def test_pi_phase_deterministic(tmp_path):
    artifact_a = tmp_path / "pi_a.json"
    artifact_b = tmp_path / "pi_b.json"
    res_a = run_pi_phase(
        rotations=1024,
        precision=1e-8,
        seed=123,
        samples=8,
        artifact_path=str(artifact_a),
    )
    res_b = run_pi_phase(
        rotations=1024,
        precision=1e-8,
        seed=123,
        samples=8,
        artifact_path=str(artifact_b),
    )
    assert res_a["summary"] == res_b["summary"]
