from __future__ import annotations

import math

from quantacap.cosmo.t1s import energy_at_1s


def test_energy_at_1s_range() -> None:
    data = energy_at_1s()
    total = data["E_total_J"]
    assert math.isfinite(total) and total > 0
    assert 1e40 <= total <= 1e54
    density = data["rho_J_m3"]
    assert math.isfinite(density) and density > 0
    ratio = data["rho_ratio_vs_modern_scaled"]
    assert ratio > 0
