from quantacap.quantum.bell import bell_counts


def test_depolarizing_reduces_visibility():
    clean = bell_counts(shots=8192, seed=424242)
    noisy = bell_counts(shots=8192, seed=424242, noise={"depol": 0.1})
    assert any(k in noisy for k in ("01", "10"))
    assert sum(noisy.values()) == 8192
    assert sum(clean.values()) == 8192
