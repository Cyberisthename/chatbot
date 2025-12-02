"""State-vector utilities supporting CPU and optional GPU backends."""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .xp import get_xp, to_numpy


def init_state(
    n: int,
    *,
    use_gpu: bool = False,
    dtype: str = "complex128",
):
    if n < 1:
        raise ValueError("number of qubits must be >= 1")
    xp = get_xp(use_gpu)
    dim = 1 << n
    dt = xp.complex128 if dtype == "complex128" else xp.complex64
    psi = xp.zeros((dim, 1), dtype=dt)
    psi[0, 0] = 1.0
    return psi, xp


def _axis_for_qubit(qubit: int, n: int) -> int:
    if qubit < 0 or qubit >= n:
        raise ValueError(f"invalid qubit index {qubit} for n={n}")
    return n - 1 - qubit


def _validate_state(psi, n: int, xp) -> Tuple[int, int]:
    dim = psi.shape[0]
    if psi.shape != (dim, 1):
        raise ValueError("state-vector must be a column vector")
    if dim != 1 << n:
        raise ValueError(f"state dimension mismatch for n={n}")
    return dim, n


def apply_unitary(psi, U, targets: Sequence[int], n: int, xp):
    targets = list(targets)
    if not targets:
        raise ValueError("apply_unitary requires at least one target qubit")
    dim, _ = _validate_state(psi, n, xp)
    m = len(targets)
    if U.shape != (1 << m, 1 << m):
        raise ValueError("unitary dimension mismatch for provided targets")

    psi_tensor = psi.reshape((2,) * n)
    axes = [_axis_for_qubit(t, n) for t in targets]
    permuted = xp.moveaxis(psi_tensor, axes, range(m))
    block = permuted.reshape(1 << m, -1)
    updated = xp.matmul(U, block).reshape((2,) * n)
    restored = xp.moveaxis(updated, range(m), axes)
    return restored.reshape(dim, 1)


def probs(psi, xp):
    flat = psi.reshape(-1)
    amplitudes = xp.abs(flat) ** 2
    return to_numpy(xp, amplitudes)


def measure_counts(psi, shots: int = 4096, seed: int = 424242, xp=None) -> dict[str, int]:
    xp = xp or np
    dim = psi.shape[0]
    n = int(math.log2(dim))
    if dim != 1 << n:
        raise ValueError("psi length must be a power of two")
    probs_np = probs(psi, xp)
    probs_np = probs_np / probs_np.sum()
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(dim, size=shots, p=probs_np)
    counts: dict[str, int] = {}
    for idx in outcomes:
        bitstring = format(int(idx), f"0{n}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def to_density_matrix(psi, xp=None):
    xp = xp or np
    rho = psi @ xp.conjugate(psi.T)
    return to_numpy(xp, rho)
