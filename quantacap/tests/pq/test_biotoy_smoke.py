"""Smoke test for pq-biotoy experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-biotoy tests")


def test_biotoy_smoke():
    from quantacap.pq.biotoy import run

    result = run(N=32, T=50, lam=0.01, seed=777, gif=False)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["N"] == 32
    assert summary["T"] == 50
    assert "psnr" in summary
    assert "energy" in summary
    assert "memory_halftime" in summary
