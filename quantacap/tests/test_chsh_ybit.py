from quantacap.experiments.chsh_ybit import run_chsh_y


def test_chsh_ybit_deterministic_violation():
    params = dict(
        shots=20000,
        depol=0.0,
        seed=424242,
        seed_id="demo.ybit",
        lam=0.85,
        eps=0.02,
        delta=0.03,
        graph_nodes=512,
        graph_out=3,
        graph_gamma=0.87,
        backend="statevector",
        use_gpu=False,
        dtype="complex128",
        chi=16,
    )
    out1 = run_chsh_y(**params)
    out2 = run_chsh_y(**params)
    assert out1["S"] > 2.2
    assert abs(out1["S"] - out2["S"]) < 1e-9
