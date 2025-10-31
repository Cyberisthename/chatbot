from __future__ import annotations

import math

import pytest

from quantacap.experiments.virtual_element import (
    combined_binding_energy,
    search_isotopes,
    weizsacker_binding_energy,
)

np = pytest.importorskip("numpy", reason="NumPy is required for virtual element tests")


def test_weizsacker_positive_binding() -> None:
    be = weizsacker_binding_energy(120, 300)
    assert math.isfinite(be) and be > 0


def test_combined_metrics_fields() -> None:
    metrics = combined_binding_energy(120, 300)
    assert metrics["binding_energy_MeV"] > 0
    assert metrics["binding_energy_per_A_MeV"] > 0
    assert "stability_score" in metrics
    assert math.isfinite(metrics["stability_score"])


def test_search_shape_and_mc() -> None:
    out = search_isotopes(Z_min=110, Z_max=111, A_min_offset=150, A_max_offset=151, n_mc=3, seed=1)
    entries = out["entries"]
    assert len(entries) == 4
    # ensure Monte Carlo statistics present and non-zero variance for at least one entry
    assert any(row.get("score_std", 0.0) > 0 for row in entries)
    matrix = out["stability_matrix"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 2
