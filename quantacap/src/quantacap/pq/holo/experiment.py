"""Holographic entropy computing toy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _binary_entropy(p: float) -> float:
    """Shannon entropy for binary distribution."""
    if p in (0.0, 1.0) or p < 0.0 or p > 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def run(
    N: int = 64,
    samples: int = 50,
    seed: int = 424242,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run holographic entropy experiment."""
    from quantacap.utils.seed import set_seed

    set_seed(seed, np)
    rng = np.random.default_rng(seed)

    artifacts_dir = Path("artifacts/pq/holo")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Generate random 3D voxel fields
    field = rng.integers(0, 2, size=(N, N, N))

    max_radius = max(2, N // 2 - 1)
    if max_radius <= 2:
        radii = np.array([2])
    else:
        radii = np.linspace(2, max_radius, samples, dtype=int)
    radii = np.unique(radii[radii > 1])

    entropy_values = []
    areas = []

    for r in radii:
        low = int(r)
        high = int(N - r)
        if high <= low:
            continue
        center = rng.integers(low, high, size=3)
        sub = field[
            center[0] - r : center[0] + r,
            center[1] - r : center[1] + r,
            center[2] - r : center[2] + r,
        ]

        entropy_values.append(float(_binary_entropy(np.mean(sub))))

        # Surface area of cube region (with side length 2r)
        side = 2 * r
        area = 6 * (side ** 2)
        areas.append(float(area))

    if not areas:
        entropy_values = [float(_binary_entropy(np.mean(field)))]
        areas = [float(6 * (N ** 2))]

    entropy_values = np.array(entropy_values, dtype=float)
    areas = np.array(areas, dtype=float)

    # Linear fit to compute holographic ratio
    if np.all(areas == 0):
        k = 0.0
        residuals = entropy_values
    else:
        k = float(np.dot(areas, entropy_values) / np.dot(areas, areas))
        residuals = entropy_values - k * areas

    ss_tot = float(np.sum((entropy_values - np.mean(entropy_values)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    holo_ratio = float(np.mean(entropy_values / (k * areas + 1e-9)))
    residual_std = float(np.std(residuals))

    summary = {
        "N": N,
        "samples": samples,
        "samples_processed": int(len(areas)),
        "seed": seed,
        "k_fit": k,
        "r_squared": r_squared,
        "holo_ratio": holo_ratio,
        "residual_std": residual_std,
        "mean_entropy": float(np.mean(entropy_values)),
        "mean_area": float(np.mean(areas)),
    }

    summary_path = artifacts_dir / "holo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plots = {
        "entropy_vs_area": _plot_entropy_vs_area(areas, entropy_values, k, artifacts_dir),
        "density_slice": _plot_density_slice(field, artifacts_dir),
    }

    result = {
        "summary_path": str(summary_path),
        "plot_paths": {k: str(v) if v else None for k, v in plots.items()},
        "metrics": {
            "k_fit": k,
            "r_squared": r_squared,
            "holo_ratio": holo_ratio,
        },
    }

    return result


def _plot_entropy_vs_area(
    areas: np.ndarray,
    entropy_values: np.ndarray,
    k: float,
    artifacts_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(areas, entropy_values, color='tab:cyan', alpha=0.7, label='Samples')
    ax.plot(areas, k * areas, color='tab:red', label='Fit')
    ax.set_xlabel('Boundary area A(r)')
    ax.set_ylabel('Entropy H(r) [bits]')
    ax.set_title('Entropy vs area (holographic scaling)')
    ax.legend()

    path = artifacts_dir / "H_vs_area.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_density_slice(field: np.ndarray, artifacts_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    slice_idx = field.shape[2] // 2
    slice_data = field[:, :, slice_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(slice_data, cmap='inferno', origin='lower')
    ax.set_title('Voxel density slice (mid-plane)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)

    path = artifacts_dir / "density_slice.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
