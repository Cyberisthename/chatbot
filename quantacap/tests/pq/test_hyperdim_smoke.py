"""Smoke test for pq-hyperdim experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-hyperdim tests")


def test_hyperdim_smoke():
    from quantacap.pq.hyperdim import run

    result = run(N=10, chi=8, depth=6, seed=99)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["N"] == 10
    assert summary["chi"] == 8
    assert "overlap" in summary
    assert "memory_bytes" in summary
