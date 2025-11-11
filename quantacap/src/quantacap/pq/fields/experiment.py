"""Sub-quantum field computing experiment via complex field interference."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def run(
    N: int = 256,
    T: int = 400,
    src: int = 2,
    seed: int = 424242,
    gif: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run sub-quantum field computing experiment.
    
    Models computing with continuous complex fields on 2D spacetime grid
    using interference as logic. Uses a simplified Complex Ginzburg-Landau
    toy model with linear wave updates and controlled sources.
    
    Args:
        N: Grid size (NxN)
        T: Number of time steps
        src: Number of sources injecting phases
        seed: Random seed
        gif: Generate evolution GIF if True
        **kwargs: Additional parameters
        
    Returns:
        Summary dictionary with paths and metrics
    """
    from quantacap.utils.seed import set_seed
    set_seed(seed, np)
    
    artifacts_dir = Path("artifacts/pq/fields")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize complex field
    phi = np.zeros((N, N), dtype=np.complex128)
    
    # Place sources at random locations
    rng = np.random.RandomState(seed)
    source_locs = []
    source_phases = []
    for i in range(src):
        x = rng.randint(N // 4, 3 * N // 4)
        y = rng.randint(N // 4, 3 * N // 4)
        source_locs.append((x, y))
        source_phases.append(2 * np.pi * rng.rand())
    
    # Place detectors
    det_locs = [
        (N // 4, N // 4),
        (3 * N // 4, N // 4),
        (N // 4, 3 * N // 4),
        (3 * N // 4, 3 * N // 4),
    ]
    
    # Evolution parameters
    dt = 0.01
    diffusion = 0.5
    frames = []
    detector_readings = []
    
    frame_stride = max(1, T // 100)
    x_grid, y_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    gaussian_scale = 2 * (N / 20) ** 2
    
    # Time evolution using wave equation with sources
    for t in range(T):
        # Add source terms
        for (sx, sy), phase in zip(source_locs, source_phases):
            amplitude = 0.1 * np.exp(1j * (phase + 0.05 * t))
            dist_sq = (x_grid - sx) ** 2 + (y_grid - sy) ** 2
            source_profile = amplitude * np.exp(-dist_sq / gaussian_scale)
            phi += source_profile * dt
        
        # Laplacian for diffusion/wave propagation
        phi_padded = np.pad(phi, 1, mode="constant")
        laplacian = (
            phi_padded[:-2, 1:-1] + phi_padded[2:, 1:-1] +
            phi_padded[1:-1, :-2] + phi_padded[1:-1, 2:] -
            4 * phi
        )
        
        # Update: simple diffusion with rotation
        phi += dt * diffusion * laplacian
        phi *= np.exp(-0.001 * dt)  # Small damping
        
        # Record detector intensities
        det_intensities = [np.abs(phi[x, y]) ** 2 for x, y in det_locs]
        detector_readings.append(det_intensities)
        
        # Save frames for GIF
        if gif and t % frame_stride == 0:
            frames.append(np.abs(phi) ** 2)
    
    # Compute metrics
    detector_readings = np.array(detector_readings)
    
    # Visibility at each detector
    visibilities = []
    for i in range(len(det_locs)):
        I = detector_readings[:, i]
        I_max = np.max(I)
        I_min = np.min(I)
        if I_max + I_min > 1e-12:
            vis = (I_max - I_min) / (I_max + I_min)
        else:
            vis = 0.0
        visibilities.append(float(vis))
    
    mean_visibility = float(np.mean(visibilities))
    
    # Mutual information (simplified): discretize detector patterns
    # Treat each detector as binary based on threshold
    threshold = np.median(detector_readings)
    det_binary = (detector_readings > threshold).astype(int)
    
    # Calculate entropy for mutual information proxy
    def entropy(x: np.ndarray) -> float:
        unique, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    
    # Mutual information between input pattern (sources) and output (detectors)
    # Simplified: entropy of detector patterns
    joint_states = [''.join(map(str, row)) for row in det_binary]
    mi = entropy(np.array(joint_states))
    
    # Total energy
    energy = float(np.sum(np.abs(phi) ** 2))
    
    # Save artifacts
    summary = {
        "N": N,
        "T": T,
        "sources": src,
        "seed": seed,
        "visibility_mean": mean_visibility,
        "visibilities": visibilities,
        "mutual_information_bits": mi,
        "final_energy": energy,
        "detector_locations": det_locs,
        "source_locations": source_locs,
    }
    
    summary_path = artifacts_dir / "fields_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save final field
    field_path = artifacts_dir / "fields_last.npy"
    np.save(field_path, phi)
    
    # Generate interference plot
    plot_path = _plot_interference(phi, det_locs, source_locs, artifacts_dir)
    
    # Generate GIF if requested
    gif_path = None
    if gif and frames:
        gif_path = _make_gif(frames, artifacts_dir)
    
    result = {
        "summary_path": str(summary_path),
        "field_path": str(field_path),
        "plot_path": str(plot_path) if plot_path else None,
        "gif_path": str(gif_path) if gif_path else None,
        "metrics": {
            "visibility": mean_visibility,
            "mutual_information": mi,
            "energy": energy,
        },
    }
    
    return result


def _plot_interference(
    phi: np.ndarray,
    det_locs: list,
    source_locs: list,
    artifacts_dir: Path,
) -> Path | None:
    """Generate interference pattern plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    intensity = np.abs(phi) ** 2
    im = ax.imshow(intensity.T, origin='lower', cmap='viridis', interpolation='bilinear')
    
    # Mark sources
    for idx, (x, y) in enumerate(source_locs):
        ax.plot(x, y, "r*", markersize=15, label="Source" if idx == 0 else "")
    
    # Mark detectors
    for idx, (x, y) in enumerate(det_locs):
        ax.plot(
            x,
            y,
            "wo",
            markersize=10,
            markeredgecolor="red",
            markeredgewidth=2,
            label="Detector" if idx == 0 else "",
        )
    
    plt.colorbar(im, ax=ax, label='Intensity |φ|²')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Field Interference Pattern (Final)')
    ax.legend()
    
    plot_path = artifacts_dir / "fields_interf.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def _make_gif(frames: list, artifacts_dir: Path) -> Path | None:
    """Generate evolution GIF."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except (ImportError, RuntimeError):
        return None
    
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(np.max(f) for f in frames)
    
    im = ax.imshow(frames[0].T, origin='lower', cmap='plasma', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.set_title('Field Evolution')
    
    def update(frame_idx: int) -> tuple:
        im.set_data(frames[frame_idx].T)
        return (im,)
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    
    gif_path = artifacts_dir / "fields_evolution.gif"
    anim.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    
    return gif_path
