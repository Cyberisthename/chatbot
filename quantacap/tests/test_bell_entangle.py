from quantacap.quantum.bell import bell_counts


def test_bell_clean_correlations():
    counts = bell_counts(shots=8192, seed=424242)
    keys = set(counts.keys())
    assert keys.issubset({"00", "11"})
    assert sum(counts.values()) == 8192
