"""Quantum reversal fidelity sweep using the circuit backend."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence
from quantacap.quantum.gates import CNOT, H, RX, RY, RZ
from quantacap.quantum.statevector import apply_unitary, init_state
from quantacap.quantum.xp import to_numpy
from quantacap.utils.optional_import import optional_import


def _np():
    return optional_import("numpy", pip_name="numpy", purpose="quantum reversal sweep")


@dataclass
class ReversalResult:
    noise_levels: List[float]
    fidelities: List[float]
    guard_triggered: bool
    parameters: Dict[str, float]


def _single_qubit_noise(noise: float, xp) -> object:
    if noise <= 0:
        return xp.eye(2, dtype=xp.complex128)
    axis = xp.asarray([1.0, 1.0, 1.0])
    axis = axis / xp.linalg.norm(axis)
    angle = noise
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return xp.array(
        [
            [c - 1j * axis[2] * s, (-1j * axis[0] - axis[1]) * s],
            [(-1j * axis[0] + axis[1]) * s, c + 1j * axis[2] * s],
        ],
        dtype=xp.complex128,
    )


def run_quantum_reversal(
    *,
    n_qubits: int = 3,
    depth: int = 12,
    noise_levels: Sequence[float] | None = None,
    seed: int = 424242,
    guard_threshold: float = 0.05,
) -> ReversalResult:
    """Apply a random circuit followed by an attempted noisy reversal."""

    if n_qubits <= 0 or depth <= 0:
        raise ValueError("n_qubits and depth must be positive")

    np = _np()
    rng = np.random.default_rng(seed)

    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1]
    noise_schedule = list(noise_levels)

    psi, xp_backend = init_state(n_qubits)
    xp_module = xp_backend
    ops: List[tuple] = []

    for _ in range(depth):
        if rng.random() < 0.7:
            gate_choice = rng.choice(["H", "RX", "RY", "RZ"])
            target = int(rng.integers(0, n_qubits))
            if gate_choice == "H":
                gate = H(xp=xp_module)
            elif gate_choice == "RX":
                gate = RX(rng.normal(math.pi / 4, 0.2), xp=xp_module)
            elif gate_choice == "RY":
                gate = RY(rng.normal(-math.pi / 5, 0.2), xp=xp_module)
            else:
                gate = RZ(rng.normal(math.pi / 6, 0.1), xp=xp_module)
            psi = apply_unitary(psi, gate, [target], n_qubits, xp_module)
            ops.append((gate, [target]))
        else:
            if n_qubits < 2:
                continue
            control = int(rng.integers(0, n_qubits - 1))
            target = (control + 1) % n_qubits
            gate = CNOT(xp=xp_module)
            psi = apply_unitary(psi, gate, [control, target], n_qubits, xp_module)
            ops.append((gate, [control, target]))

    fidelities: List[float] = []
    guard_triggered = False

    for noise in noise_schedule:
        psi_rev = psi.copy()
        for gate, targets in reversed(ops):
            inv = xp_module.conjugate(xp_module.transpose(gate))
            if noise > 0:
                if len(targets) == 1:
                    noise_gate = _single_qubit_noise(noise, xp_module)
                else:
                    noise_gate = xp_module.kron(
                        _single_qubit_noise(noise, xp_module),
                        _single_qubit_noise(noise, xp_module),
                    )
                inv = noise_gate @ inv
            psi_rev = apply_unitary(psi_rev, inv, targets, n_qubits, xp_module)

        init_state_vec, _ = init_state(n_qubits)
        v0 = to_numpy(xp_module, init_state_vec).reshape(-1)
        vf = to_numpy(xp_module, psi_rev).reshape(-1)
        fidelity = float(abs(np.vdot(v0, vf)) ** 2)
        fidelities.append(fidelity)
        if fidelity < guard_threshold:
            guard_triggered = True
            break

    return ReversalResult(
        noise_levels=noise_schedule[: len(fidelities)],
        fidelities=fidelities,
        guard_triggered=guard_triggered,
        parameters={
            "n_qubits": float(n_qubits),
            "depth": float(depth),
            "seed": float(seed),
            "guard_threshold": float(guard_threshold),
        },
    )
