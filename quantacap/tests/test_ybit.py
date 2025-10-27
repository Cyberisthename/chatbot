import math

from quantacap.primitives.ybit import YBit
from quantacap.primitives.zbit import ZBit


def test_ybit_adjustment_and_phase():
    alpha = 1.0 / math.sqrt(2)
    beta = 1j / math.sqrt(2)
    zbit = ZBit(seed=9876)
    y1 = YBit((alpha, beta), zbit, lam=0.85, eps_phase=0.02)
    y2 = YBit((alpha, beta), zbit, lam=0.85, eps_phase=0.02)
    assert math.isclose(y1.adjusted_prob_1(), y2.adjusted_prob_1())
    assert 0.0 <= y1.adjusted_prob_1() <= 1.0
    assert 0.0 <= y1.phase_nudge() < 2.0 * math.pi * 0.02 + 1e-12
