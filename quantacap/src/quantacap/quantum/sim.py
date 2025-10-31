import numpy as np, math


def h_gate():
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], complex)


def x_gate():
    return np.array([[0, 1], [1, 0]], complex)


def run_hadamard(seed=42, shots=1024):
    np.random.seed(seed)
    psi = np.array([1, 0], complex)
    psi = h_gate() @ psi
    p = np.abs(psi) ** 2
    counts = {"0": 0, "1": 0}
    for _ in range(shots):
        outcome = "0" if np.random.rand() < p[0] else "1"
        counts[outcome] += 1
    return dict(probs=p.tolist(), counts=counts)


