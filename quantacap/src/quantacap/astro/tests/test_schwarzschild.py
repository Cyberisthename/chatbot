from __future__ import annotations

from quantacap.astro.schwarzschild import integrate_null_geodesic
from quantacap.astro.lensing import render_lensing_map, run_lens_atom_equivalence
from quantacap.experiments.atom1d import run_atom2d_transition


def test_deflection_monotonic() -> None:
    low = integrate_null_geodesic(8.0, steps=500)
    high = integrate_null_geodesic(12.0, steps=500)
    assert abs(low.deflection) > abs(high.deflection)


def test_lensing_map_meta(tmp_path) -> None:
    meta = render_lensing_map(resolution=8, impact_min=3.0, impact_max=5.0, artifact_prefix=str(tmp_path / "ring"))
    assert meta["resolution"] == 8


def test_lens_atom_equivalence(tmp_path) -> None:
    atom = run_atom2d_transition(
        n=4,
        L=3.0,
        sigma_primary=0.9,
        sigma_secondary=1.1,
        separation=0.8,
        adapter_id="atom2d.equiv",
        pi_adapter_id=None,
    )
    report = run_lens_atom_equivalence(
        resolution=8,
        impact_min=3.0,
        impact_max=5.0,
        atom_artifact=atom["artifact"],
        adapter_id="astro.equiv.test",
    )
    assert "correlation" in report
