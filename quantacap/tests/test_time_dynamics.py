from __future__ import annotations

import math

import pytest

from quantacap.experiments.material_time import simulate_material_time
from quantacap.experiments.timecrystal import run_time_crystal
from quantacap.experiments.timerev import run_quantum_reversal

np = pytest.importorskip("numpy", reason="NumPy required for time-dynamics tests")


def test_timecrystal_autocorr_periodicity() -> None:
    result = run_time_crystal(N=6, steps=40, disorder=0.05, jitter=0.02, seed=1)
    assert pytest.approx(result.autocorrelation[0], rel=1e-6) == 1.0
    assert result.detected is True
    assert result.subharmonic_strength >= 0.0


def test_material_time_monotonic_clock() -> None:
    result = simulate_material_time(traps=16, steps=200, rate=5e-3, temperature=0.4, seed=2)
    diffs = [b - a for a, b in zip(result.material_time, result.material_time[1:])]
    assert all(delta >= 0 for delta in diffs)
    assert result.reversal_attempts > 0


def test_quantum_reversal_perfect_case() -> None:
    out = run_quantum_reversal(n_qubits=2, depth=6, noise_levels=[0.0], seed=3)
    assert len(out.fidelities) == 1
    assert pytest.approx(out.fidelities[0], rel=1e-9) == 1.0
    assert out.guard_triggered is False
