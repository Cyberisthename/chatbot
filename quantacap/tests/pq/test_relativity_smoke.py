"""Smoke test for pq-relativity experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-relativity tests")


def test_relativity_smoke():
    from quantacap.pq.relativity import run

    result = run(nodes=16, edges=48, beta=0.5, seed=123)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["nodes"] == 16
    assert "speedup" in summary
    assert "gamma_mean" in summary
    assert summary["causality_violations"] == 0
