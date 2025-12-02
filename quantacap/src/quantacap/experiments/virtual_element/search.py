"""Search grid utilities for virtual element discovery."""
from __future__ import annotations

import time
from typing import Dict, List, Mapping

from quantacap.core.adapter_store import create_adapter
from quantacap.utils.optional_import import optional_import

from .models import DEFAULT_WEIZSACKER_PARAMS, combined_binding_energy


def _np():
    return optional_import(
        "numpy", pip_name="numpy", purpose="virtual element search"
    )


def search_isotopes(
    *,
    Z_min: int = 100,
    Z_max: int = 130,
    A_min_offset: int = 150,
    A_max_offset: int = 220,
    n_mc: int = 0,
    seed: int = 424242,
    adapter_top_k: int = 5,
    adapter_prefix: str = "virtual.element",
    params: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, object]:
    """Enumerate synthetic isotopes and compute heuristic stability metrics."""

    if Z_min > Z_max:
        raise ValueError("Z_min must be <= Z_max")
    if A_min_offset > A_max_offset:
        raise ValueError("A_min_offset must be <= A_max_offset")

    np = _np()
    rng = np.random.default_rng(seed)
    entries: List[Dict[str, float]] = []

    start = time.perf_counter()
    for Z in range(Z_min, Z_max + 1):
        for offset in range(A_min_offset, A_max_offset + 1):
            A = Z + offset
            metrics = combined_binding_energy(Z, A, params)
            entry: Dict[str, float | int | None] = {
                "Z": int(Z),
                "A": int(A),
                "N": int(A - Z),
                "binding_energy_MeV": metrics["binding_energy_MeV"],
                "binding_energy_per_A_MeV": metrics["binding_energy_per_A_MeV"],
                "Q_alpha_MeV": metrics["Q_alpha_MeV"],
                "sf_vulnerability": metrics["sf_vulnerability"],
                "stability_score": metrics["stability_score"],
                "delta_shell_MeV": metrics["delta_shell_MeV"],
            }
            if n_mc > 0:
                samples = []
                for _ in range(n_mc):
                    perturb = {
                        "weizsacker": {
                            key: rng.normal(val, 0.03 * abs(val) + 1e-3)
                            for key, val in DEFAULT_WEIZSACKER_PARAMS.items()
                        },
                        "shell": {
                            "strength": rng.normal(5.0, 0.5),
                            "width_Z": rng.normal(6.0, 0.8),
                            "width_N": rng.normal(8.0, 1.0),
                        },
                    }
                    sample_metrics = combined_binding_energy(Z, A, perturb)
                    samples.append(sample_metrics["stability_score"])
                entry["score_mean"] = float(np.mean(samples))
                entry["score_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
            entries.append({k: float(v) if isinstance(v, float) else v for k, v in entry.items()})

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    entries.sort(key=lambda item: item["stability_score"], reverse=True)

    top = entries[: adapter_top_k if adapter_top_k > 0 else 0]
    for record in top:
        adapter_id = f"{adapter_prefix}.Z{record['Z']}.A{record['A']}"
        create_adapter(
            adapter_id,
            data={
                "Z": record["Z"],
                "A": record["A"],
                "N": record["N"],
                "binding_energy_MeV": record["binding_energy_MeV"],
                "binding_energy_per_A_MeV": record["binding_energy_per_A_MeV"],
                "Q_alpha_MeV": record["Q_alpha_MeV"],
                "sf_vulnerability": record["sf_vulnerability"],
                "stability_score": record["stability_score"],
                "score_mean": record.get("score_mean"),
                "score_std": record.get("score_std"),
            },
        )

    Z_values = sorted({row["Z"] for row in entries})
    A_values = sorted({row["A"] for row in entries})
    matrix = np.full((len(Z_values), len(A_values)), np.nan)
    for row in entries:
        i = Z_values.index(row["Z"])
        j = A_values.index(row["A"])
        matrix[i, j] = row["stability_score"]

    return {
        "entries": entries,
        "top": top,
        "Z_values": Z_values,
        "A_values": A_values,
        "stability_matrix": matrix.tolist(),
        "latency_ms": elapsed_ms,
    }
