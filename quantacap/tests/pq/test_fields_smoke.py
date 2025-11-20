"""Smoke test for pq-fields experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-fields tests")


def test_fields_smoke():
    from quantacap.pq.fields import run

    result = run(N=32, T=20, src=2, seed=42, gif=False)

    assert "summary_path" in result
    assert "metrics" in result

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert "visibility_mean" in summary
    assert "mutual_information_bits" in summary
    assert summary["N"] == 32
    assert summary["T"] == 20
    assert summary["seed"] == 42
