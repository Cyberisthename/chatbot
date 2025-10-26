from __future__ import annotations

import math

import numpy as np

from quantacap.experiments.chsh_rehearsal import run_chsh_scan


def test_chsh_rehearsal_scan_basic():
    result = run_chsh_scan(
        pmin=0.0,
        pmax=0.2,
        steps=5,
        shots=20000,
        adapter_id="test.chsh.scan",
        seed=424242,
    )
    ps = np.linspace(0.0, 0.2, 5)
    assert result["p"] == list(ps)
    assert result["S_clean"] > 2.5
    noisy = np.array(result["S_noisy"])
    diffs = np.diff(noisy)
    assert float(diffs.mean()) <= 0.0
    assert math.isclose(result["S_rehearsed"][0], result["S_clean"], rel_tol=1e-3)
