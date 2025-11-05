"""Hyperdimensional tensor network computing experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


Operation = Tuple[str, int, np.ndarray]


def run(
    N: int = 48,
    chi: int = 32,
    depth: int = 40,
    seed: int = 424242,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the hyperdimensional tensor network computing experiment."""
    from quantacap.utils.seed import set_seed

    set_seed(seed, np)
    rng = np.random.default_rng(seed)

    artifacts_dir = Path("artifacts/pq/hyperdim")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    circuit = _generate_circuit(N, depth, rng)

    ref_N = min(N, 10)
    ref_circuit = _filter_circuit(circuit, ref_N)
    psi_dense = _simulate_dense(ref_N, ref_circuit)

    chi_values = sorted({max(2, min(chi, x)) for x in (4, 8, chi)})
    chi_data = []

    for chi_val in chi_values:
        mps_ref, _ = _simulate_mps(ref_N, ref_circuit, chi_val)
        psi_mps = _mps_to_dense(mps_ref)
        overlap = float(np.abs(np.vdot(psi_dense, psi_mps)) ** 2)
        chi_data.append((chi_val, overlap))

    start = time.perf_counter()
    mps_full, stats = _simulate_mps(N, circuit, chi)
    runtime = time.perf_counter() - start

    overlap = chi_data[-1][1]
    memory_bytes = int(sum(tensor.nbytes for tensor in mps_full))

    summary = {
        "N": N,
        "chi": chi,
        "depth": depth,
        "seed": seed,
        "overlap": overlap,
        "memory_bytes": memory_bytes,
        "runtime_seconds": runtime,
        "bond_dims": stats["bond_dims"],
    }

    summary_path = artifacts_dir / "hyperdim_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = _plot_accuracy_vs_chi(chi_data, artifacts_dir)

    result = {
        "summary_path": str(summary_path),
        "plot_path": str(plot_path) if plot_path else None,
        "metrics": {
            "overlap": overlap,
            "memory_bytes": memory_bytes,
            "runtime_seconds": runtime,
        },
    }

    return result


def _generate_circuit(N: int, depth: int, rng: np.random.Generator) -> List[Operation]:
    circuit: List[Operation] = []
    for layer in range(depth):
        # Single-qubit rotations
        for site in range(N):
            circuit.append(("single", site, _random_single(rng)))
        # Two-qubit entanglers on even bonds
        for site in range(0, N - 1, 2):
            circuit.append(("two", site, _random_two(rng)))
        # Two-qubit entanglers on odd bonds
        for site in range(1, N - 1, 2):
            circuit.append(("two", site, _random_two(rng)))
    return circuit


def _random_single(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    lam = rng.uniform(0, 2 * np.pi)
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    )


def _random_two(rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(mat)
    diag = np.diag(r)
    phase = diag / np.abs(diag)
    return q * phase


def _filter_circuit(circuit: List[Operation], N: int) -> List[Operation]:
    filtered = []
    for kind, site, gate in circuit:
        if kind == "single" and site < N:
            filtered.append((kind, site, gate))
        elif kind == "two" and site + 1 < N:
            filtered.append((kind, site, gate))
    return filtered


def _simulate_dense(N: int, circuit: List[Operation]) -> np.ndarray:
    psi = np.zeros(2 ** N, dtype=np.complex128)
    psi[0] = 1.0

    for kind, site, gate in circuit:
        if kind == "single":
            psi = _apply_single_dense(psi, gate, site, N)
        else:
            psi = _apply_two_dense(psi, gate, site, N)
    return psi


def _apply_single_dense(psi: np.ndarray, gate: np.ndarray, site: int, N: int) -> np.ndarray:
    psi = psi.reshape([2] * N)
    psi = np.tensordot(psi, gate, axes=([site], [0]))
    psi = np.moveaxis(psi, -1, site)
    return psi.reshape(-1)


def _apply_two_dense(psi: np.ndarray, gate: np.ndarray, site: int, N: int) -> np.ndarray:
    psi = psi.reshape([2] * N)
    gate_tensor = gate.reshape(2, 2, 2, 2)
    psi = np.tensordot(psi, gate_tensor, axes=([site, site + 1], [0, 1]))
    psi = np.moveaxis(psi, [-2, -1], [site, site + 1])
    return psi.reshape(-1)


def _simulate_mps(N: int, circuit: List[Operation], chi: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    mps = [np.zeros((1, 2, 1), dtype=np.complex128) for _ in range(N)]
    for tensor in mps:
        tensor[0, 0, 0] = 1.0  # |0> state
    bond_dims = []

    for kind, site, gate in circuit:
        if kind == "single":
            mps[site] = _apply_single_mps(mps[site], gate)
        else:
            mps[site], mps[site + 1] = _apply_two_mps(mps[site], mps[site + 1], gate, chi)
            bond_dims.append(int(mps[site].shape[2]))
    stats = {"bond_dims": bond_dims}
    return mps, stats


def _apply_single_mps(tensor: np.ndarray, gate: np.ndarray) -> np.ndarray:
    new_tensor = np.tensordot(gate, tensor, axes=([1], [1]))
    return np.transpose(new_tensor, (1, 0, 2))


def _apply_two_mps(
    left: np.ndarray,
    right: np.ndarray,
    gate: np.ndarray,
    chi: int,
) -> Tuple[np.ndarray, np.ndarray]:
    l_dim, _, mid_dim = left.shape
    r_mid, _, r_dim = right.shape

    theta = np.tensordot(left, right, axes=(2, 0))  # (l, 2, 2, r)
    theta = np.transpose(theta, (0, 3, 1, 2))  # (l, r, 2, 2)
    theta = theta.reshape(l_dim * r_dim, 4)
    theta = theta @ gate.T
    theta = theta.reshape(l_dim, r_dim, 2, 2)
    theta = np.transpose(theta, (0, 2, 3, 1))  # (l, 2, 2, r)
    theta = theta.reshape(l_dim * 2, r_dim * 2)

    U, S, Vh = np.linalg.svd(theta, full_matrices=False)
    chi_eff = min(chi, len(S))

    U = U[:, :chi_eff]
    S = S[:chi_eff]
    Vh = Vh[:chi_eff, :]

    left_new = U.reshape(l_dim, 2, chi_eff)
    right_new = (S[:, None] * Vh).reshape(chi_eff, r_dim, 2)
    right_new = np.transpose(right_new, (0, 2, 1))

    return left_new, right_new


def _mps_to_dense(mps: List[np.ndarray]) -> np.ndarray:
    state = mps[0][0]
    state = state.reshape(1, 2, mps[0].shape[2])

    tensor = mps[0]
    psi = tensor
    for tensor in mps[1:]:
        psi = np.tensordot(psi, tensor, axes=(psi.ndim - 1, 0))
    psi = psi.reshape(-1)
    return psi


def _plot_accuracy_vs_chi(
    chi_data: List[Tuple[int, float]],
    artifacts_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    chis, overlaps = zip(*chi_data)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(chis, overlaps, marker='o', color='tab:blue')
    ax.set_xlabel('Bond dimension χ')
    ax.set_ylabel('Overlap with dense reference')
    ax.set_title('Accuracy vs bond dimension')

    path = artifacts_dir / "accuracy_vs_chi.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
