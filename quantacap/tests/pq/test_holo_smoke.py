"""Smoke test for pq-holo experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-holo tests")


def test_holo_smoke():
    from quantacap.pq.holo import run

    result = run(N=24, samples=8, seed=11)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["N"] == 24
    assert summary["samples"] == 8
    assert "k_fit" in summary
    assert "r_squared" in summary
