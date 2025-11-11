"""Smoke test for pq-topo experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="NumPy is required for pq-topo tests")


def test_topo_smoke():
    from quantacap.pq.topo import run

    result = run(braid="s1 s2^-1", shots=256, noise=0.02, seed=7)

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["braid"] == "s1 s2^-1"
    assert "fidelity" in summary
    assert "topo_stability" in summary
    assert summary["shots"] == 256
