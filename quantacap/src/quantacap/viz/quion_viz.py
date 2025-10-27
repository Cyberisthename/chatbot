"""Plotting helpers for Quion++ visualisations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import numpy as np

from quantacap.quion.state import mags_phases, pca2_project, stack_re_im
from quantacap.utils.optional_import import optional_import


def _get_mpl() -> object:
    """Load matplotlib with a headless backend when available."""

    mpl = optional_import(
        "matplotlib",
        pip_name="matplotlib",
        purpose="render Quion++ figures",
    )
    try:
        mpl.use("Agg", force=True)
    except Exception:
        try:
            mpl.use("Agg")
        except Exception:
            pass
    return mpl


def _get_pyplot():
    _get_mpl()
    return optional_import(
        "matplotlib.pyplot",
        pip_name="matplotlib",
        purpose="render Quion++ figures",
    )


def _get_animation():
    _get_mpl()
    return optional_import(
        "matplotlib.animation",
        pip_name="matplotlib",
        purpose="render Quion++ animations",
    )


def _ensure_path(path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_quion_frame(
    psi: np.ndarray,
    out_png: Path | str,
    *,
    title: str = "Quion++",
    history: Sequence[np.ndarray] | None = None,
    meta: dict | None = None,
) -> dict:
    """Render a single frame and return summary statistics."""
    plt = _get_pyplot()
    path = _ensure_path(out_png)
    mags, phases = mags_phases(psi)
    summary = {
        "sum_mags": float(np.sum(mags)),
        "max_mag": float(np.max(mags)),
        "phases": phases.tolist(),
    }
    if meta:
        summary.update({f"meta_{k}": v for k, v in meta.items()})

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title)

    ax0 = fig.add_subplot(2, 2, 1)
    bars = ax0.bar(range(len(mags)), mags, color=plt.cm.viridis(mags))
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title("Magnitude Simplex")
    ax0.set_xticks(range(len(mags)))
    ax0.set_ylabel("|psi|^2")
    for bar, mag in zip(bars, mags):
        ax0.text(bar.get_x() + bar.get_width() / 2, mag + 0.01, f"{mag:.2f}", ha="center", va="bottom", fontsize=8)

    ax1 = fig.add_subplot(2, 2, 2, polar=True)
    ax1.set_title("Phase Wheel")
    spokes = np.linspace(0.0, 2 * math.pi, len(phases), endpoint=False)
    ax1.scatter(spokes, np.ones_like(spokes), c=phases, cmap="twilight", s=80)
    for angle, phase in zip(spokes, phases):
        ax1.plot([angle, angle], [0, 1], color="lightgray", linewidth=0.8)
        ax1.text(angle, 1.05, f"{phase:.2f}", ha="center", va="center", fontsize=8)

    ax2 = fig.add_subplot(2, 1, 2)
    vecs = [stack_re_im(p) for p in history] if history else [stack_re_im(psi)]
    proj = pca2_project(np.stack(vecs))
    ax2.set_title("2D Projection")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    if len(proj) > 1:
        ax2.plot(proj[:-1, 0], proj[:-1, 1], color="lightblue", linewidth=1.5)
    ax2.scatter(proj[-1, 0], proj[-1, 1], color="crimson", s=60)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


def animate_series(
    states: Sequence[np.ndarray],
    out_path: Path | str,
    *,
    fps: int = 12,
    title: str = "Quion++ Time-lapse",
    prefer: str = "auto",
) -> Path:
    """Create an animation from state history and return final path."""
    if len(states) == 0:
        raise ValueError("animation requires at least one state")
    plt = _get_pyplot()
    animation = _get_animation()
    out = _ensure_path(out_path)

    history_vectors = [stack_re_im(s) for s in states]
    projections = pca2_project(np.stack(history_vectors))

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2)
    ax_mag = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1], polar=True)
    ax_proj = fig.add_subplot(gs[1, :])
    fig.suptitle(title)

    mags0, phases0 = mags_phases(states[0])
    bars = ax_mag.bar(range(len(mags0)), mags0, color=plt.cm.viridis(mags0))
    ax_mag.set_ylim(0.0, 1.0)
    ax_mag.set_title("Magnitude Simplex")
    ax_mag.set_ylabel("|psi|^2")

    spokes = np.linspace(0.0, 2 * math.pi, len(phases0), endpoint=False)
    phase_scatter = ax_phase.scatter(spokes, np.ones_like(spokes), c=phases0, cmap="twilight", s=80)
    ax_phase.set_title("Phase Wheel")
    for angle in spokes:
        ax_phase.plot([angle, angle], [0, 1], color="lightgray", linewidth=0.8)

    path_line, = ax_proj.plot([], [], color="lightblue", linewidth=1.5)
    point_scatter = ax_proj.scatter([], [], color="crimson", s=60)
    ax_proj.set_title("2D Projection")
    ax_proj.set_xlabel("PC1")
    ax_proj.set_ylabel("PC2")
    ax_proj.grid(True, linestyle="--", alpha=0.4)

    def _update(frame: int) -> List:
        mags, phases = mags_phases(states[frame])
        for bar, mag in zip(bars, mags):
            bar.set_height(mag)
        phase_scatter.set_array(phases)
        proj = projections[: frame + 1]
        if len(proj) > 1:
            path_line.set_data(proj[:-1, 0], proj[:-1, 1])
        point_scatter.set_offsets(proj[-1])
        return [*bars, phase_scatter, path_line, point_scatter]

    ani = animation.FuncAnimation(fig, _update, frames=len(states), blit=False)

    def _save(writer_name: str, suffix: str) -> Path:
        out_with_suffix = out.with_suffix(suffix)
        out_with_suffix.parent.mkdir(parents=True, exist_ok=True)
        writer = getattr(animation, writer_name)
        ani.save(out_with_suffix, writer=writer(fps=fps))
        return out_with_suffix

    saved: Path | None = None
    preferred = prefer
    if preferred == "auto":
        preferred = "FFMpegWriter"
    try:
        if preferred == "FFMpegWriter":
            saved = _save("FFMpegWriter", ".mp4")
        elif preferred == "PillowWriter":
            saved = _save("PillowWriter", ".gif")
        else:
            saved = _save(preferred, out.suffix or ".mp4")
    except Exception:
        fallback = "PillowWriter"
        saved = _save(fallback, ".gif")

    plt.close(fig)
    return saved
