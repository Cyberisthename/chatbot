"""
TIME TRAVEL ENGINE v1
by you & Cent

This script:
  - Simulates multiple kinds of timelines
  - Attempts to "time travel" by modifying past states
  - Checks if the future changes (simulated causality violation)

Modes:
  A) Simple logic timeline
  B) Quantum timeline (1-qubit)
  C) Multiverse branches
  D) Chaos timeline (butterfly effect)

Requires: numpy
    pip install numpy
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy


# ======================================================
# COMMON UTILITIES
# ======================================================


def divider(title: str | None = None) -> None:
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60 + "\n")


# ======================================================
# MODE A: SIMPLE LOGIC TIMELINE
# ======================================================


def simple_evolve(value: int) -> int:
    """
    Simple deterministic rule:
        v_{t+1} = (3*v + 7) mod 100
    """
    return (3 * value + 7) % 100


def run_simple_timeline(steps: int = 20, initial_value: int = 10, travel_time: int = 5) -> bool:
    """
    - Build a timeline of integer values
    - Then 'time travel' from final time back to travel_time:
        we overwrite timeline[travel_time] with a different value
    - Recompute the future from that point and see if it changed.
    """
    divider("MODE A: SIMPLE LOGIC TIMELINE")

    timeline = [initial_value]
    for _ in range(steps - 1):
        timeline.append(simple_evolve(timeline[-1]))

    print(f"Initial value at t=0   : {initial_value}")
    print(f"Original final state   : {timeline[-1]}")
    print(f"Original timeline      : {timeline}")

    timeline_tt = timeline.copy()
    past_value_original = timeline_tt[travel_time]
    past_value_new = (past_value_original * 11 + 23) % 100
    timeline_tt[travel_time] = past_value_new

    for t in range(travel_time + 1, steps):
        timeline_tt[t] = simple_evolve(timeline_tt[t - 1])

    print("\n[Time travel attempt]")
    print(f"Past time index       : {travel_time}")
    print(f"Original past value   : {past_value_original}")
    print(f"New past value        : {past_value_new}")
    print(f"New final state       : {timeline_tt[-1]}")
    print(f"New timeline          : {timeline_tt}")

    changed = timeline_tt[-1] != timeline[-1]

    if changed:
        print("\n✅ SIMPLE MODE: Changing the past changed the future inside the sim.")
    else:
        print("\n⚠️ SIMPLE MODE: Future remained the same (no effect).")

    return changed


# ======================================================
# MODE B: QUANTUM TIMELINE (1-QUBIT)
# ======================================================


sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def normalize(psi: np.ndarray) -> np.ndarray:
    return psi / np.linalg.norm(psi)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hamiltonian(omega_x: float, omega_y: float, omega_z: float) -> np.ndarray:
    return 0.5 * (omega_x * sigma_x + omega_y * sigma_y + omega_z * sigma_z)


def time_evolution_operator(H: np.ndarray, dt: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(H)
    exp_diag = np.diag(np.exp(-1j * eigvals * dt))
    V = eigvecs
    Vinv = np.linalg.inv(V)
    return V @ exp_diag @ Vinv


def orthogonal_vector(v: np.ndarray) -> np.ndarray:
    v0, v1 = v
    if abs(v0) < 0.9:
        w = np.array([1 + 0j, 0 + 0j])
    else:
        w = np.array([0 + 0j, 1 + 0j])
    w = w - np.vdot(v, w) * v
    return normalize(w)


def unitary_mapping(psi_from: np.ndarray, psi_to: np.ndarray) -> np.ndarray:
    v1 = normalize(psi_from)
    u1 = normalize(psi_to)
    v2 = orthogonal_vector(v1)
    u2 = orthogonal_vector(u1)
    V = np.column_stack([v1, v2])
    U = np.column_stack([u1, u2])
    return U @ V.conj().T


def run_quantum_timeline(
    steps: int = 30,
    dt: float = 0.2,
    jump_from_step: int = 22,
    jump_to_step: int = 7,
) -> bool:
    divider("MODE B: QUANTUM TIMELINE (1-QUBIT)")

    omega_x, omega_y, omega_z = 0.4, 0.7, 1.1
    H = hamiltonian(omega_x, omega_y, omega_z)
    U_dt = time_evolution_operator(H, dt)

    psi0 = normalize(np.array([1 + 0j, 0 + 0j]))

    states = [psi0]
    for _ in range(1, steps):
        states.append(U_dt @ states[-1])
    states = [normalize(s) for s in states]

    psi_early = states[jump_to_step]
    psi_late = states[jump_from_step]

    print(f"Total steps        : {steps}")
    print(f"dt                 : {dt}")
    print(f"Jump FROM step tb  : {jump_from_step}")
    print(f"Jump TO   step ta  : {jump_to_step}")
    print()
    print("State at early time (ta):", psi_early)
    print("State at late  time (tb):", psi_late)

    U_jump = unitary_mapping(psi_late, psi_early)
    mapped = U_jump @ psi_late
    F_map = fidelity(psi_early, mapped)

    print("\n[Time travel attempt in Hilbert space]")
    print("Mapped state at tb ->", mapped)
    print(f"Fidelity(early, mapped) = {F_map:.12f}")

    success = F_map > 0.999999

    if success:
        print("\n✅ QUANTUM MODE: You achieved microscopic time travel of the qubit state.")
    else:
        print("\n⚠️ QUANTUM MODE: Mapping not perfect.")

    return success


# ======================================================
# MODE C: MULTIVERSE TIMELINE
# ======================================================


def multiverse_evolve(state: int) -> list[int]:
    """
    Simple branching rule:
      next_state_1 = (a + 1) mod 10
      next_state_2 = (a * 2 + 3) mod 10
    Each branch is just an integer label.
    """
    a = state
    return [(a + 1) % 10, (a * 2 + 3) % 10]


def run_multiverse_timeline(depth: int = 5, initial_state: int = 3, travel_depth: int = 3) -> bool:
    divider("MODE C: MULTIVERSE TIMELINE")

    levels = [[initial_state]]
    for _ in range(depth):
        next_level = []
        for s in levels[-1]:
            next_level.extend(multiverse_evolve(s))
        levels.append(next_level)

    print("Original multiverse tree:")
    for d, lvl in enumerate(levels):
        print(f"Depth {d}: {lvl}")
    original_final = levels[-1].copy()

    if travel_depth >= len(levels):
        travel_depth = len(levels) - 2

    branch_before = levels[travel_depth][0]
    altered_branch = (branch_before + 5) % 10
    levels_tt = deepcopy(levels)
    levels_tt[travel_depth][0] = altered_branch

    for d in range(travel_depth + 1, depth + 1):
        new_level = []
        if d == travel_depth + 1:
            for s in levels_tt[travel_depth]:
                new_level.extend(multiverse_evolve(s))
        else:
            for s in levels_tt[d - 1]:
                new_level.extend(multiverse_evolve(s))
        levels_tt[d] = new_level

    print("\n[Time travel attempt: altering one branch in the past]")
    print(f"Travel depth           : {travel_depth}")
    print(f"Original branch value  : {branch_before}")
    print(f"Altered branch value   : {altered_branch}")
    print("\nNew multiverse tree:")
    for d, lvl in enumerate(levels_tt):
        print(f"Depth {d}: {lvl}")

    new_final = levels_tt[-1]
    changed = new_final != original_final

    if changed:
        print("\n✅ MULTIVERSE MODE: The branch alteration changed the final branch distribution.")
    else:
        print("\n⚠️ MULTIVERSE MODE: No net change at final level (weirdly stable).")

    return changed


# ======================================================
# MODE D: CHAOS TIMELINE (BUTTERFLY EFFECT)
# ======================================================


def logistic_step(x: float, r: float = 3.9) -> float:
    return r * x * (1 - x)


def run_chaos_timeline(steps: int = 40, x0: float = 0.2, epsilon: float = 1e-6) -> bool:
    divider("MODE D: CHAOS TIMELINE (BUTTERFLY EFFECT)")

    x = x0
    path_original = [x]
    for _ in range(steps - 1):
        x = logistic_step(x)
        path_original.append(x)

    x = x0 + epsilon
    path_perturbed = [x]
    for _ in range(steps - 1):
        x = logistic_step(x)
        path_perturbed.append(x)

    print(f"Initial x0             : {x0}")
    print(f"Perturbed x0           : {x0 + epsilon}")
    print(f"Steps                  : {steps}")
    print("\nLast few values (original vs perturbed):")
    for i in range(steps - 5, steps):
        print(
            f"t={i:2d} | original={path_original[i]:.10f}  "
            f"perturbed={path_perturbed[i]:.10f}"
        )

    diff_final = abs(path_original[-1] - path_perturbed[-1])

    print(f"\nFinal difference |Δx|   = {diff_final:.10f}")

    success = diff_final > 1e-2

    if success:
        print("\n✅ CHAOS MODE: Tiny change in the 'past' exploded into a big difference.")
    else:
        print("\n⚠️ CHAOS MODE: System not chaotic enough under these params.")

    return success


# ======================================================
# MAIN DRIVER
# ======================================================


def main() -> None:
    divider("TIME TRAVEL ENGINE v1")

    results: dict[str, bool] = {}

    results["simple"] = run_simple_timeline(
        steps=20, initial_value=13, travel_time=6
    )

    results["quantum"] = run_quantum_timeline(
        steps=30, dt=0.25, jump_from_step=22, jump_to_step=7
    )

    results["multiverse"] = run_multiverse_timeline(
        depth=5, initial_state=3, travel_depth=3
    )

    results["chaos"] = run_chaos_timeline(steps=40, x0=0.2, epsilon=1e-8)

    divider("SUMMARY")
    for mode, ok in results.items():
        print(f"{mode.upper():10s} : {'TIME EFFECT ACHIEVED' if ok else 'NO EFFECT'}")

    print("\nNote: all of this is simulated time travel / causality violation.")
    print("You just built a multi-mode time travel *simulation* engine.")


if __name__ == "__main__":
    main()
