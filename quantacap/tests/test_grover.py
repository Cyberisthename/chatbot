from quantacap.quantum.grover import grover_search


def test_grover_n3_marked5_success():
    out = grover_search(n=3, marked_index=5, shots=4096, seed=424242)
    # Expect >= 0.9 success probability for n=3 with optimal iterations
    assert out["iters"] >= 1
    assert out["success_prob"] >= 0.9, out
