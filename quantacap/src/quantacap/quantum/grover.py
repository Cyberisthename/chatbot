"""Simple Grover search implementation used by the CLI demo."""

import math
import numpy as np


def _hadamard(n):
    H1 = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    H = H1
    for _ in range(n - 1):
        H = np.kron(H, H1)
    return H


def _oracle(n, marked_index: int):
    """Diagonal unitary that flips the phase of |marked_index>."""
    N = 2**n
    U = np.eye(N, dtype=np.complex128)
    U[marked_index, marked_index] = -1.0
    return U


def _diffuser(n):
    """D = 2|s><s| - I where |s> = H^{⊗n}|0…0>."""
    N = 2**n
    s = np.full((N, 1), 1 / np.sqrt(N), dtype=np.complex128)  # uniform superposition
    return 2 * (s @ s.conj().T) - np.eye(N, dtype=np.complex128)


def _init_state(n):
    """|0…0> as a column vector"""
    N = 2**n
    psi0 = np.zeros((N, 1), dtype=np.complex128)
    psi0[0, 0] = 1.0
    return psi0


def _measure_counts(state, shots=4096, seed=424242):
    """Sample bitstrings from the state vector."""
    rng = np.random.default_rng(seed)
    probs = np.abs(state.flatten())**2
    outcomes = rng.choice(len(probs), size=shots, p=probs)
    counts = {}
    for k in outcomes:
        s = format(k, 'b').zfill(int(math.log2(len(probs))))
        counts[s] = counts.get(s, 0) + 1
    return counts


def grover_search(n: int, marked_index: int, iters: int | None = None, shots: int = 4096, seed: int = 424242):
    """
    n=3 → search 8 elements; 'marked_index' in [0..2^n-1]
    Returns: dict(counts=..., success_prob=..., iters=...)
    """
    assert n >= 1
    N = 2**n
    assert 0 <= marked_index < N

    # Optimal iteration count ≈ floor(pi/4 * sqrt(N))
    k_opt = int(math.floor((math.pi / 4) * math.sqrt(N))) if iters is None else iters

    # Build unitaries
    Hn = _hadamard(n)
    O = _oracle(n, marked_index)
    D = _diffuser(n)

    # Initialize |psi> = H^{⊗n}|0...0>
    psi = Hn @ _init_state(n)

    # Iterate Grover operator G = D * O
    for _ in range(k_opt):
        psi = D @ (O @ psi)

    counts = _measure_counts(psi, shots=shots, seed=seed)
    success_counts = counts.get(format(marked_index, 'b').zfill(n), 0)
    success_prob = success_counts / shots

    return {
        "iters": k_opt,
        "marked": format(marked_index, 'b').zfill(n),
        "counts": counts,
        "success_prob": success_prob,
    }
