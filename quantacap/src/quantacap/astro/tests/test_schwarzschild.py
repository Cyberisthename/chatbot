from __future__ import annotations

from quantacap.astro.schwarzschild import integrate_null_geodesic
from quantacap.astro.lensing import render_lensing_map


def test_deflection_monotonic() -> None:
    low = integrate_null_geodesic(8.0, steps=500)
    high = integrate_null_geodesic(12.0, steps=500)
    assert abs(low.deflection) > abs(high.deflection)


def test_lensing_map_meta(tmp_path) -> None:
    meta = render_lensing_map(resolution=8, impact_min=3.0, impact_max=5.0, artifact_prefix=str(tmp_path / "ring"))
    assert meta["resolution"] == 8
