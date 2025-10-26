import math

from quantacap.primitives.ggraph import GGraph


def test_ggraph_influence_deterministic_and_gamma_scaling():
    g_low = GGraph(n=128, out_degree=3, gamma=0.75, seed=1234)
    g_high = GGraph(n=128, out_degree=3, gamma=0.92, seed=1234)
    eta_low = g_low.influence("demo")
    eta_low_repeat = g_low.influence("demo")
    eta_high = g_high.influence("demo")

    assert all(math.isclose(a, b) for a, b in zip(eta_low, eta_low_repeat))
    assert all(0.0 <= v <= 1.0 for v in eta_low)
    assert all(0.0 <= v <= 1.0 for v in eta_high)
    mean_low = sum(eta_low) / 2.0
    mean_high = sum(eta_high) / 2.0
    assert mean_high >= mean_low
