"""Topological braid-based logic experiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def run(
    braid: str = "s1 s2^-1 s1",
    shots: int = 8192,
    noise: float = 0.03,
    seed: int = 424242,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the topological braid logic experiment."""
    from quantacap.utils.seed import set_seed

    set_seed(seed, np)
    rng = np.random.default_rng(seed)

    artifacts_dir = Path("artifacts/pq/topo")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Parse braid word
    generators = _parse_braid(braid)
    braid_length = len(generators)

    # Build clean unitary and apply to initial state |psi0>
    sigma_mats = _generators()
    unitary = np.eye(2, dtype=np.complex128)
    for g in generators:
        unitary = sigma_mats[g] @ unitary

    psi0 = np.array([1 / math.sqrt(2), 1j / math.sqrt(2)], dtype=np.complex128)
    psi_clean = unitary @ psi0

    # Shots with noise perturbation and projection back to braid class
    fidelities = []
    unitary_deviation = []
    counts = {"0": 0, "1": 0}
    for _ in range(shots):
        perturbed = _apply_noise(unitary, noise, rng)
        # Project back by normalizing via QR to unitary form
        q, _ = np.linalg.qr(perturbed)
        psi_noisy = q @ psi0
        fidelity = float(np.abs(np.vdot(psi_clean, psi_noisy)) ** 2)
        fidelities.append(fidelity)
        unitary_deviation.append(float(np.linalg.norm(unitary - q) / np.sqrt(unitary.size)))

        probs = np.abs(psi_noisy) ** 2
        outcome = rng.choice(["0", "1"], p=probs)
        counts[outcome] += 1

    fidelity_mean = float(np.mean(fidelities))
    topo_stability = float(max(0.0, 1.0 - np.mean(unitary_deviation)))

    summary = {
        "braid": braid,
        "braid_length": braid_length,
        "shots": shots,
        "noise": noise,
        "fidelity": fidelity_mean,
        "topo_stability": topo_stability,
        "counts": counts,
        "seed": seed,
    }

    summary_path = artifacts_dir / "topo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    unitary_path = artifacts_dir / "unitary.npy"
    np.save(unitary_path, unitary)

    plot_paths = {
        "braid_plot": _plot_braid_paths(generators, artifacts_dir),
        "histogram": _plot_histogram(counts, artifacts_dir),
    }

    result = {
        "summary_path": str(summary_path),
        "unitary_path": str(unitary_path),
        "plot_paths": {k: str(v) if v else None for k, v in plot_paths.items()},
        "metrics": {
            "fidelity": fidelity_mean,
            "topo_stability": topo_stability,
            "braid_length": braid_length,
        },
    }

    return result


def _parse_braid(braid: str) -> List[str]:
    tokens = braid.strip().split()
    cleaned = []
    for t in tokens:
        if "^-1" in t:
            cleaned.append(t.replace("^-1", "_inv"))
        else:
            cleaned.append(t)
    return cleaned


def _generators() -> Dict[str, np.ndarray]:
    phase = np.exp(-1j * np.pi / 4)
    sigma1 = phase * np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    sigma2 = phase * np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    sigma1_inv = np.linalg.inv(sigma1)
    sigma2_inv = np.linalg.inv(sigma2)

    return {
        "s1": sigma1,
        "s2": sigma2,
        "s1_inv": sigma1_inv,
        "s2_inv": sigma2_inv,
    }


def _apply_noise(unitary: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    delta = (rng.normal(size=unitary.shape) + 1j * rng.normal(size=unitary.shape)) * noise
    return unitary + delta


def _plot_braid_paths(generators: List[str], artifacts_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    n_anyons = 3
    steps = len(generators)

    xs = np.zeros((n_anyons, steps + 1))
    for i in range(n_anyons):
        xs[i, 0] = i

    for t, gen in enumerate(generators, start=1):
        xs[:, t] = xs[:, t - 1]
        if gen.startswith('s1'):
            xs[0, t] = xs[1, t - 1]
            xs[1, t] = xs[0, t - 1]
        elif gen.startswith('s2'):
            xs[1, t] = xs[2, t - 1]
            xs[2, t] = xs[1, t - 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    time = np.arange(steps + 1)
    for i in range(n_anyons):
        ax.plot(time, xs[i], label=f"Anyon {i}")

    ax.set_xlabel('Time step')
    ax.set_ylabel('Worldline index')
    ax.set_title('Braid Worldlines')
    ax.legend()

    path = artifacts_dir / "braid_plot.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_histogram(counts: Dict[str, int], artifacts_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, values, color=['tab:purple', 'tab:orange'])
    ax.set_ylabel('Counts')
    ax.set_title('Measurement Outcomes')

    path = artifacts_dir / "histogram.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
