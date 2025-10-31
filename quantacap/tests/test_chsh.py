from quantacap.experiments.chsh import run_chsh


def test_chsh_violation():
    out = run_chsh(shots=20000, depol=0.0, seed=424242)
    assert out["S"] > 2.5
