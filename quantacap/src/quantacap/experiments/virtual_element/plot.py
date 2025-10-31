"""Plot helpers for the virtual element search (optional dependencies)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from quantacap.utils.optional_import import optional_import


def _get_plotter():
    mpl = optional_import(
        "matplotlib",
        pip_name="matplotlib",
        purpose="render virtual element plots",
    )
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433 (runtime import)

    return mpl, plt


def plot_heatmap(
    Z_values: Sequence[int],
    A_values: Sequence[int],
    matrix,
    *,
    out_path: os.PathLike[str] | str,
    title: str = "Virtual stability heatmap",
) -> Path | None:
    try:
        _, plt = _get_plotter()
    except RuntimeError:
        return None

    import numpy as np

    grid = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(A_values)))
    ax.set_xticklabels(A_values, rotation=45, ha="right")
    ax.set_yticks(range(len(Z_values)))
    ax.set_yticklabels(Z_values)
    ax.set_xlabel("Mass number A")
    ax.set_ylabel("Proton number Z")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Stability score (a.u.)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_binding_curve(results: Sequence[dict], *, Z: int, out_path: os.PathLike[str] | str) -> Path | None:
    try:
        _, plt = _get_plotter()
    except RuntimeError:
        return None

    filtered = [row for row in results if row["Z"] == Z]
    if not filtered:
        return None
    filtered.sort(key=lambda row: row["A"])
    A_vals = [row["A"] for row in filtered]
    be_vals = [row["binding_energy_per_A_MeV"] for row in filtered]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(A_vals, be_vals, marker="o", label=f"Z={Z}")
    ax.set_xlabel("Mass number A")
    ax.set_ylabel("Binding energy per nucleon (MeV)")
    ax.set_title("Binding curve for Z=%d" % Z)
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_top_candidates(results: Sequence[dict], *, out_path: os.PathLike[str] | str, top_k: int = 10) -> Path | None:
    try:
        _, plt = _get_plotter()
    except RuntimeError:
        return None

    top = sorted(results, key=lambda row: row["stability_score"], reverse=True)[:top_k]
    if not top:
        return None

    labels = [f"Z{row['Z']}A{row['A']}" for row in top]
    scores = [row["stability_score"] for row in top]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, scores, color="tab:purple")
    ax.set_ylabel("Stability score (a.u.)")
    ax.set_title("Top virtual isotopes")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
