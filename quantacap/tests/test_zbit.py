import math

from quantacap.primitives.zbit import ZBit


def test_zbit_forbidden_band_and_determinism():
    zb1 = ZBit(seed=12345)
    zb2 = ZBit(seed=12345)
    assert math.isclose(zb1.value(), zb2.value())
    assert not (1.0 <= zb1.value() <= 2.0)
    bias = zb1.bias_sigma()
    assert 0.0 <= bias <= 1.0
    phase = zb1.phase()
    assert 0.0 <= phase < 2.0 * math.pi
