#!/usr/bin/env python
"""
Demo: Quantum Time Reversal

Creates a qubit state, evolves it with a sequence of unitary gates, applies the
exact reverse evolution, and verifies that the initial state is recovered.

Run from the repository root:
    python quantacap/examples/demo_qubit_time_reversal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sure we can import quantacap when executed directly from the repo.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantacap.quantum import gates
from quantacap.quantum.statevector import apply_unitary, init_state
from quantacap.quantum.xp import to_numpy


def dagger(U, xp):
    """Hermitian conjugate of a unitary matrix for the current backend."""
    return xp.transpose(xp.conjugate(U))


def compose_unitary(steps, xp, dtype):
    """Compose the total unitary from a list of (label, matrix) steps."""
    U_total = xp.eye(2, dtype=dtype)
    for _, gate in steps:
        U_total = gate @ U_total
    return U_total


def format_state(state, xp):
    """Return a flattened numpy array for pretty printing."""
    return to_numpy(xp, state).reshape(-1)


def time_reversal_experiment():
    # Initialize |0> on a single qubit (column vector of size 2).
    psi0, xp = init_state(1, use_gpu=False, dtype="complex128")

    print("Initial state |psi0>:")
    print(format_state(psi0, xp))
    print()

    # Define the forward evolution sequence.
    steps = [
        ("H", gates.H(xp=xp)),
        ("Rz(0.7)", gates.RZ(0.7, xp=xp)),
        ("X", gates.X(xp=xp)),
        ("Rz(-1.3)", gates.RZ(-1.3, xp=xp)),
        ("H", gates.H(xp=xp)),
    ]

    # Apply the gates sequentially.
    psi_forward = psi0
    for label, gate in steps:
        psi_forward = apply_unitary(psi_forward, gate, [0], 1, xp)

    print("After forward evolution |psi_forward>:")
    print(format_state(psi_forward, xp))
    print()

    # Compose the total unitary and build its Hermitian conjugate.
    U_total = compose_unitary(steps, xp, psi0.dtype)
    U_reverse = dagger(U_total, xp)

    # Apply the perfect time-reversal.
    psi_reversed = apply_unitary(psi_forward, U_reverse, [0], 1, xp)

    print("After time reversal |psi_reversed>:")
    print(format_state(psi_reversed, xp))
    print()

    # Fidelity check between the initial and reversed states.
    overlap = xp.vdot(psi0.reshape(-1), psi_reversed.reshape(-1))
    fidelity = xp.abs(overlap) ** 2
    fidelity_val = float(to_numpy(xp, xp.asarray(fidelity)))

    print(f"Fidelity between initial and reversed state: {fidelity_val:.12f}")
    if fidelity_val > 0.999999:
        print("✅ For this qubit, 'time' was basically perfectly reversed.")
    else:
        print("⚠️ Not a perfect reversal — something messed with the evolution.")


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    time_reversal_experiment()
