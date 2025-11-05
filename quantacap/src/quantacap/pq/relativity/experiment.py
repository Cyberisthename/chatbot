"""Relativistic time dilation computing toy."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def run(
    nodes: int = 64,
    edges: int = 256,
    beta: float = 0.6,
    seed: int = 424242,
    graph: str | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run relativistic computing experiment."""
    from quantacap.utils.seed import set_seed

    set_seed(seed, np)
    rng = np.random.default_rng(seed)

    artifacts_dir = Path("artifacts/pq/relativity")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dag = _generate_graph(nodes, edges, rng)
    gamma_values = _gamma_profile(nodes, beta, rng)

    time_newton, path_newton = _longest_path(dag, nodes, weight=lambda u, v, w: w)
    time_rel, path_rel = _longest_path(
        dag, nodes, weight=lambda u, v, w: w / gamma_values[u]
    )

    speedup = float(time_newton[-1] / time_rel[-1]) if time_rel[-1] > 0 else math.inf
    critical_path = path_rel[-1]

    summary = {
        "nodes": nodes,
        "edges": len(dag)
    }

    summary.update(
        {
            "beta": beta,
            "gamma_mean": float(np.mean(gamma_values)),
            "gamma_max": float(np.max(gamma_values)),
            "gamma_min": float(np.min(gamma_values)),
            "speedup": speedup,
            "critical_path": critical_path,
            "time_newton": float(time_newton[-1]),
            "time_relativistic": float(time_rel[-1]),
            "causality_violations": int(np.sum(time_rel < 0)),
            "seed": seed,
        }
    )

    summary_path = artifacts_dir / "relativity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plots = {
        "timing_hist": _plot_timings(time_newton, time_rel, artifacts_dir),
        "graph_plot": _plot_graph(dag, nodes, critical_path, artifacts_dir),
    }

    result = {
        "summary_path": str(summary_path),
        "plot_paths": {k: str(v) if v else None for k, v in plots.items()},
        "metrics": {
            "speedup": speedup,
            "gamma_mean": float(np.mean(gamma_values)),
            "causality_violations": int(np.sum(time_rel < 0)),
        },
    }

    return result


def _generate_graph(nodes: int, edges: int, rng: np.random.Generator) -> List[Tuple[int, int, float]]:
    edges = min(edges, nodes * (nodes - 1) // 2)
    selected = set()
    data = []
    for _ in range(edges):
        u = int(rng.integers(0, nodes - 1))
        v = int(rng.integers(u + 1, nodes))
        while (u, v) in selected:
            u = int(rng.integers(0, nodes - 1))
            v = int(rng.integers(u + 1, nodes))
        weight = float(rng.uniform(0.5, 2.0))
        selected.add((u, v))
        data.append((u, v, weight))
    return data


def _gamma_profile(nodes: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    velocities = rng.uniform(0.0, beta, size=nodes)
    velocities = np.clip(velocities, 0, 0.999)
    gamma = 1.0 / np.sqrt(1.0 - velocities ** 2)
    return gamma


def _longest_path(dag: List[Tuple[int, int, float]], nodes: int, weight) -> Tuple[np.ndarray, List[List[int]]]:
    times = np.zeros(nodes)
    path = [[i] for i in range(nodes)]

    # adjacency list by target
    adjacency = [[] for _ in range(nodes)]
    for u, v, w in dag:
        adjacency[v].append((u, w))

    for v in range(nodes):
        candidates = []
        for u, w in adjacency[v]:
            t = times[u] + weight(u, v, w)
            candidates.append((t, path[u] + [v]))
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            times[v] = best[0]
            path[v] = best[1]
        elif v > 0:
            times[v] = times[v - 1]
            path[v] = path[v - 1] + [v]
    return times, path


def _plot_timings(time_newton: np.ndarray, time_rel: np.ndarray, artifacts_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_newton, label='Newtonian time', color='tab:blue')
    ax.plot(time_rel, label='Relativistic proper time', color='tab:red')
    ax.set_xlabel('Node index')
    ax.set_ylabel('Time')
    ax.set_title('Completion time profiles')
    ax.legend()

    path = artifacts_dir / "timing_hist.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_graph(
    dag: List[Tuple[int, int, float]],
    nodes: int,
    critical_path: List[int],
    artifacts_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError):
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    xs = np.linspace(0, 1, nodes)
    ys = np.sin(xs * np.pi)

    ax.scatter(xs, ys, color='black')
    for (u, v, w) in dag:
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color='gray', alpha=0.3)

    crit_coords = [(xs[i], ys[i]) for i in critical_path]
    ax.plot([c[0] for c in crit_coords], [c[1] for c in crit_coords], color='tab:green', linewidth=2)

    ax.set_title('Relativistic task graph (critical path highlighted)')
    ax.set_xticks([])
    ax.set_yticks([])

    path = artifacts_dir / "graph.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
