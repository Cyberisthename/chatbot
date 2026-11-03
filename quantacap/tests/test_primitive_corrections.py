import math
import numpy as np
import pytest
from quantacap.primitives.ggraph import GGraph
from quantacap.primitives.ybit import YBit
from quantacap.primitives.zbit import ZBit
from quantacap.experiments.chsh_ybit import _local_amplitudes

def test_zbit_properties_and_determinism():
    for seed in range(100):
        a, b = ZBit(seed=seed), ZBit(seed=seed)
        assert a.value() == b.value()
        assert not (1 <= a.value() <= 2)
        assert 0 <= a.bias_sigma() <= 1
        assert 0 <= a.phase() < 2 * math.pi

def test_invalid_zbit_inputs():
    with pytest.raises(ValueError): ZBit(seed=1, side="middle")
    with pytest.raises(ValueError): ZBit(seed=1).bias_sigma(-1)

def test_ybit_normalization_and_bounds():
    y = YBit((3 + 0j, 4 + 0j), ZBit(seed=2), lam=0.8)
    assert y.base_prob_1() == pytest.approx(0.64)
    assert 0 <= y.adjusted_prob_1() <= 1
    assert math.isfinite(y.phase_nudge())

def test_ybit_rejects_invalid_amplitudes():
    with pytest.raises(ValueError): YBit((0j, 0j), ZBit(seed=1))
    with pytest.raises(ValueError): YBit((complex(float("nan"), 0), 1), ZBit(seed=1))

def test_ggraph_normalized_repeatable_outputs():
    graph = GGraph(n=512, out_degree=3, gamma=0.87, seed=42)
    values = [graph.influence(f"key-{i}") for i in range(100)]
    assert values == [graph.influence(f"key-{i}") for i in range(100)]
    for a, b in values:
        assert 0 <= a <= 1 and 0 <= b <= 1
        assert a + b == pytest.approx(1)

def test_ggraph_invalid_parameters():
    with pytest.raises(ValueError): GGraph(n=0)
    with pytest.raises(ValueError): GGraph(gamma=1.0)

def test_bell_state_local_marginals_are_half():
    psi = np.array([1, 0, 0, 1], dtype=np.complex128) / math.sqrt(2)
    for q in (0, 1):
        alpha, beta = _local_amplitudes(psi, q)
        assert abs(alpha)**2 == pytest.approx(0.5)
        assert abs(beta)**2 == pytest.approx(0.5)

def test_local_marginal_global_phase_invariance():
    psi = np.array([1, 0, 0, 1], dtype=np.complex128) / math.sqrt(2)
    a = _local_amplitudes(psi, 0)
    b = _local_amplitudes(psi * np.exp(1j * 0.73), 0)
    assert abs(a[0])**2 == pytest.approx(abs(b[0])**2)
    assert abs(a[1])**2 == pytest.approx(abs(b[1])**2)
