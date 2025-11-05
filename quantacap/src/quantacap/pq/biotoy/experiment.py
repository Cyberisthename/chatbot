"""BioToy dynamical field with adaptive connectivity."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def run(
    N: int = 128,
    T: int = 500,
    lam: float = 0.01,
    seed: int = 424242,
    gif: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the BioToy experiment with adaptive neural matter."""
    from quantacap.utils.seed import set_seed

    set_seed(seed, np)
    rng = np.random.default_rng(seed)

    artifacts_dir = Path("artifacts/pq/biotoy")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fields and target pattern
    phi = rng.normal(scale=0.1, size=(N, N))
    W = rng.normal(scale=0.05, size=(N, N))
    x = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    target = np.sin(np.pi * X) * np.cos(np.pi * Y)

    dt = 0.05
    diff_rate = 0.2
    hebbian_rate = 0.01
    plasticity = 0.1

    energy_trace = []
    frames = []

    frame_stride = max(1, T // 80)

    # Training phase
    for t in range(T):
        laplacian = _laplacian(phi)
        error = target - phi
        sensory_drive = plasticity * W * error
        phi += dt * (diff_rate * laplacian - 0.1 * phi + sensory_drive)
        hebbian = plasticity * (phi * target)
        W += hebbian_rate * (hebbian - lam * W)
        W = np.tanh(W)  # keep weights bounded

        energy = float(np.sum(phi ** 2) + lam * np.sum(W ** 2))
        energy_trace.append(energy)

        if gif and t % frame_stride == 0:
            frames.append(phi.copy())

    psnr = _psnr(phi, target)
    energy = float(np.mean(energy_trace[-min(25, len(energy_trace)):]))

    # Dream replay: turn off sensory drive and observe persistence
    halftime = _memory_halftime(phi, target, dt, diff_rate, rng, lam)

    summary = {
        "N": N,
        "T": T,
        "lambda": lam,
        "seed": seed,
        "psnr": psnr,
        "energy": energy,
        "memory_halftime": halftime,
    }

    summary_path = artifacts_dir / "biotoy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plots = {
        "energy_curve": _plot_energy_curve(energy_trace, artifacts_dir),
        "replay_gif": _make_replay_gif(frames, artifacts_dir) if gif else None,
    }

    result = {
        "summary_path": str(summary_path),
        "plot_paths": {k: str(v) if isinstance(v, Path) else v for k, v in plots.items()},
        "metrics": {
            "psnr": psnr,
            "energy": energy,
            "memory_halftime": halftime,
        },
    }

    return result


def _laplacian(phi: np.ndarray) -> np.ndarray:
    padded = np.pad(phi, 1, mode='wrap')
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:] -
        4 * phi
    )


def _psnr(output: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((output - target) ** 2)
    if mse <= 1e-12:
        return float('inf')
    max_i = np.max(np.abs(target))
    return float(20 * math.log10(max_i) - 10 * math.log10(mse))


def _memory_halftime(
    phi: np.ndarray,
    target: np.ndarray,
    dt: float,
    diff_rate: float,
    rng: np.random.Generator,
    lam: float,
    steps: int = 200,
) -> float:
    phi_replay = phi.copy()
    initial_corr = float(np.sum(phi * target))

    for t in range(1, steps + 1):
        laplacian = _laplacian(phi_replay)
        phi_replay += dt * (diff_rate * laplacian - 0.12 * phi_replay)
        phi_replay += rng.normal(scale=0.01, size=phi_replay.shape)

        corr = float(np.sum(phi_replay * target))
        if abs(corr) <= 0.5 * abs(initial_corr):
            return float(t * dt)

    return float(steps * dt)


def _plot_energy_curve(energy_trace: list[float], artifacts_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy_trace, color='tab:green')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Energy budget E')
    ax.set_title('BioToy energy trajectory')

    path = artifacts_dir / "energy_curve.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _make_replay_gif(frames: list[np.ndarray], artifacts_dir: Path) -> Path | None:
    if not frames:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except (ImportError, RuntimeError):
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = min(np.min(f) for f in frames)
    vmax = max(np.max(f) for f in frames)
    im = ax.imshow(frames[0], cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('BioToy replay dynamics')

    def update(idx: int):
        im.set_data(frames[idx])
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
    path = artifacts_dir / "biotoy_replay.gif"
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return path
