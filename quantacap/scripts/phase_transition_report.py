"""Phase transition analytics for π-phase experiments."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quantacap.utils.optional_import import optional_import


@dataclass
class NoiseSweep:
    seed: int
    sigma: Sequence[float]
    entropy: Sequence[float]
    steps: Sequence[int]
    plateaus: Sequence[Dict[str, float]]


@dataclass
class CouplingRun:
    kappa: float
    sync_step: float


@dataclass
class DriftRun:
    rate: float
    half_life: Optional[float]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _gather_artifacts(artifacts_dir: Path, pattern: str) -> List[Tuple[Path, Dict[str, object]]]:
    return [
        (path, _load_json(path))
        for path in sorted(artifacts_dir.glob(pattern))
        if path.is_file()
    ]


def _load_noise_runs(artifacts_dir: Path, suffix: str) -> Dict[int, NoiseSweep]:
    runs: Dict[int, NoiseSweep] = {}
    patterns = [f"pi_noise_{suffix}.json", f"pi_noise_{suffix}_*.json"]
    for pattern in patterns:
        for path, payload in _gather_artifacts(artifacts_dir, pattern):
            seed = int(payload.get("seed", 0))
            runs[seed] = NoiseSweep(
                seed=seed,
                sigma=payload.get("sigma", []),
                entropy=payload.get("entropy", []),
                steps=payload.get("entropy_steps", []),
                plateaus=payload.get("entropy_plateaus", []),
            )
    return runs


def _load_coupling_runs(artifacts_dir: Path) -> List[CouplingRun]:
    runs: List[CouplingRun] = []
    for pattern in ("pi_couple.json", "pi_couple_k_*.json"):
        for path, payload in _gather_artifacts(artifacts_dir, pattern):
            kappa = float(payload.get("kappa", 0.0))
            sync_step = float(payload.get("sync_step", 0.0))
            runs.append(CouplingRun(kappa=kappa, sync_step=sync_step))
    runs.sort(key=lambda item: item.kappa)
    return runs


def _load_drift_runs(artifacts_dir: Path) -> List[DriftRun]:
    runs: List[DriftRun] = []
    for pattern in ("pi_drift.json", "pi_drift_r_*.json"):
        for path, payload in _gather_artifacts(artifacts_dir, pattern):
            rate = float(payload.get("rate", 0.0))
            half_life = payload.get("coherence_half_life")
            runs.append(DriftRun(rate=rate, half_life=None if half_life is None else float(half_life)))
    runs.sort(key=lambda item: item.rate)
    return runs


def _bootstrap_ci(values: Sequence[float], *, seed: int = 424242, resamples: int = 1000) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    lower = float(np.quantile(means, 0.025))
    upper = float(np.quantile(means, 0.975))
    return lower, upper


def _plateau_lengths(plateaus: Iterable[Dict[str, float]]) -> List[float]:
    lengths: List[float] = []
    for entry in plateaus:
        start = entry.get("start")
        end = entry.get("end")
        if start is None or end is None:
            continue
        lengths.append(float(end) - float(start) + 1.0)
    return lengths


def _aggregate_steps(runs: Iterable[NoiseSweep]) -> Tuple[List[int], List[float]]:
    steps_set = set()
    plateaus: List[float] = []
    for run in runs:
        steps_set.update(int(step) for step in run.steps)
        plateaus.extend(_plateau_lengths(run.plateaus))
    return sorted(steps_set), plateaus


def _jaccard_distance(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


def _kneedle(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(xs[0])
    p1 = np.array([xs[0], ys[0]], dtype=float)
    p2 = np.array([xs[-1], ys[-1]], dtype=float)
    line_vec = p2 - p1
    norm = np.linalg.norm(line_vec)
    if norm == 0.0:
        return float(xs[0])
    distances = []
    for idx in range(xs.size):
        p = np.array([xs[idx], ys[idx]], dtype=float)
        distance = np.abs(np.cross(line_vec, p1 - p)) / norm
        distances.append(distance)
    max_idx = int(np.argmax(distances))
    return float(xs[max_idx])


def _second_derivative_knee(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 3:
        return float(xs[0]) if xs.size else float("nan")
    second = []
    for idx in range(1, xs.size - 1):
        val = ys[idx - 1] - 2.0 * ys[idx] + ys[idx + 1]
        second.append(val)
    min_idx = int(np.argmin(second)) + 1
    return float(xs[min_idx])


def _jackknife_ci(xs: np.ndarray, ys: np.ndarray, estimator) -> Tuple[float, Tuple[float, float]]:
    if xs.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    if xs.size <= 2:
        estimate = estimator(xs, ys)
        return estimate, (estimate, estimate)
    estimates = []
    for idx in range(xs.size):
        mask = np.ones(xs.size, dtype=bool)
        mask[idx] = False
        estimates.append(estimator(xs[mask], ys[mask]))
    estimates_arr = np.array(estimates, dtype=float)
    theta_hat = float(estimator(xs, ys))
    mean_jack = estimates_arr.mean()
    variance = (len(estimates_arr) - 1) / len(estimates_arr) * np.sum((estimates_arr - mean_jack) ** 2)
    std_err = math.sqrt(max(variance, 0.0))
    ci = (theta_hat - 1.96 * std_err, theta_hat + 1.96 * std_err)
    return theta_hat, ci


def _fit_drift(points: List[DriftRun]) -> Tuple[float, Tuple[float, float], float, List[Tuple[float, float]]]:
    filtered = [(run.rate, run.half_life) for run in points if run.rate > 0 and run.half_life]
    if len(filtered) < 2:
        if filtered:
            rate, life = filtered[0]
            return 0.0, (0.0, 0.0), 0.0, [(rate, life)]
        return float("nan"), (float("nan"), float("nan")), float("nan"), []
    rates = np.array([item[0] for item in filtered], dtype=float)
    half = np.array([item[1] for item in filtered], dtype=float)
    logx = np.log(rates)
    logy = np.log(half)
    slope, intercept = np.polyfit(logx, logy, 1)
    y_pred = slope * logx + intercept
    ss_res = float(np.sum((logy - y_pred) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    alpha = -float(slope)
    n = rates.size
    if n > 2:
        s_err = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
        x_mean = float(logx.mean())
        sxx = float(np.sum((logx - x_mean) ** 2))
        if sxx > 0:
            slope_std = s_err / math.sqrt(sxx)
            alpha_std = float(slope_std)
            ci = (alpha - 1.96 * alpha_std, alpha + 1.96 * alpha_std)
        else:
            ci = (alpha, alpha)
    else:
        ci = (alpha, alpha)
    curve = list(zip(rates.tolist(), half.tolist()))
    return alpha, ci, r2, curve


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_plot_noise(
    out_path: Path,
    up_runs: Sequence[NoiseSweep],
    down_runs: Sequence[NoiseSweep],
) -> Optional[Path]:
    if not up_runs and not down_runs:
        return None
    try:
        mpl = optional_import("matplotlib", pip_name="matplotlib", purpose="generate phase transition plots")
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in up_runs:
        ax.plot(run.sigma, run.entropy, label=f"up seed {run.seed}", color="tab:blue", alpha=0.6)
        for step in run.steps:
            ax.axvline(run.sigma[min(step, len(run.sigma) - 1)], color="tab:blue", linestyle="--", alpha=0.3)
    for run in down_runs:
        ax.plot(run.sigma, run.entropy, label=f"down seed {run.seed}", color="tab:orange", alpha=0.6)
        for step in run.steps:
            ax.axvline(run.sigma[min(step, len(run.sigma) - 1)], color="tab:orange", linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("σ (rad)")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy vs σ with detected steps")
    ax.legend(loc="best", fontsize="small")

    inset = fig.add_axes([0.6, 0.55, 0.3, 0.3])
    up_lengths = _plateau_lengths(p for run in up_runs for p in run.plateaus)
    down_lengths = _plateau_lengths(p for run in down_runs for p in run.plateaus)
    if up_lengths:
        inset.hist(up_lengths, bins=min(10, len(up_lengths)), alpha=0.6, label="up", color="tab:blue")
    if down_lengths:
        inset.hist(down_lengths, bins=min(10, len(down_lengths)), alpha=0.6, label="down", color="tab:orange")
    inset.set_title("Plateau lengths")
    inset.legend(fontsize="x-small")

    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _maybe_plot_frontier(out_path: Path, curve: List[Tuple[float, float]], knees: Dict[str, float]) -> Optional[Path]:
    if not curve:
        return None
    try:
        mpl = optional_import("matplotlib", pip_name="matplotlib", purpose="generate phase transition plots")
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return None

    xs = [pt[0] for pt in curve]
    ys = [pt[1] for pt in curve]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", color="tab:green")
    ax.set_xlabel("κ")
    ax.set_ylabel("Sync step")
    ax.set_title("Synchronization frontier")
    for label, value in knees.items():
        if math.isnan(value):
            continue
        ax.axvline(value, linestyle="--", label=label)
    ax.legend(fontsize="small")
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _maybe_plot_drift(out_path: Path, curve: List[Tuple[float, float]], alpha: float) -> Optional[Path]:
    if not curve:
        return None
    try:
        mpl = optional_import("matplotlib", pip_name="matplotlib", purpose="generate phase transition plots")
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return None

    rates = np.array([pt[0] for pt in curve])
    half = np.array([pt[1] for pt in curve])
    fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(6, 7))
    ax_main.loglog(rates, half, marker="s", color="tab:red", linestyle="none")
    coeff = np.polyfit(np.log(rates), np.log(half), 1)
    fit = np.exp(coeff[1]) * rates ** coeff[0]
    ax_main.loglog(rates, fit, color="black", label=f"fit α≈{alpha:.3f}")
    ax_main.set_xlabel("Drift rate")
    ax_main.set_ylabel("Coherence half-life")
    ax_main.legend(fontsize="small")

    residuals = np.log(half) - (coeff[0] * np.log(rates) + coeff[1])
    ax_resid.plot(rates, residuals, marker="o")
    ax_resid.set_xscale("log")
    ax_resid.set_ylabel("Log residual")
    ax_resid.set_xlabel("Drift rate")

    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(
    *,
    artifacts_dir: str | Path = "artifacts",
    output_prefix: str | Path | None = None,
    enable_plots: bool = True,
) -> Dict[str, object]:
    base = Path(artifacts_dir)
    _ensure_dir(base)
    prefix = Path(output_prefix) if output_prefix else base / "phase_transition"
    summary_path = base / "phase_transition_summary.json"
    bundle_path = base / "phase_transition_bundle.zip"

    noise_up = _load_noise_runs(base, "up")
    noise_down = _load_noise_runs(base, "down")

    steps_up, plateaus_up = _aggregate_steps(noise_up.values())
    steps_down, plateaus_down = _aggregate_steps(noise_down.values())
    jaccard = _jaccard_distance(steps_up, steps_down)
    up_counts = [len(run.steps) for run in noise_up.values()]
    down_counts = [len(run.steps) for run in noise_down.values()]
    up_ci = _bootstrap_ci(up_counts)
    down_ci = _bootstrap_ci(down_counts)

    coupling = _load_coupling_runs(base)
    xs = np.array([run.kappa for run in coupling], dtype=float)
    ys = np.array([run.sync_step for run in coupling], dtype=float)
    knees: Dict[str, float] = {}
    if xs.size:
        knee_kneedle, knee_ci = _jackknife_ci(xs, ys, _kneedle)
        second_knee, second_ci = _jackknife_ci(xs, ys, _second_derivative_knee)
        knees = {
            "kneedle": knee_kneedle,
            "kneedle_ci_lower": knee_ci[0],
            "kneedle_ci_upper": knee_ci[1],
            "second_derivative": second_knee,
            "second_derivative_ci_lower": second_ci[0],
            "second_derivative_ci_upper": second_ci[1],
        }
    curve = [(run.kappa, run.sync_step) for run in coupling]

    drift_runs = _load_drift_runs(base)
    alpha, alpha_ci, r2, drift_curve = _fit_drift(drift_runs)

    summary: Dict[str, object] = {
        "hysteresis": {
            "steps_up": steps_up,
            "steps_down": steps_down,
            "jaccard": jaccard,
            "plateaus": {"up": plateaus_up, "down": plateaus_down},
            "step_count_ci_up": list(up_ci) if up_ci else None,
            "step_count_ci_down": list(down_ci) if down_ci else None,
        },
        "frontier": {
            "knee_kneedle": knees.get("kneedle", float("nan")),
            "knee_kneedle_ci": [
                knees.get("kneedle_ci_lower", float("nan")),
                knees.get("kneedle_ci_upper", float("nan")),
            ],
            "knee_second_deriv": knees.get("second_derivative", float("nan")),
            "knee_second_deriv_ci": [
                knees.get("second_derivative_ci_lower", float("nan")),
                knees.get("second_derivative_ci_upper", float("nan")),
            ],
            "sync_curve": curve,
        },
        "drift": {
            "alpha": alpha,
            "alpha_ci": list(alpha_ci),
            "R2": r2,
            "points": drift_curve,
        },
    }

    cosmo_path = base / "early_universe_t1s.json"
    if cosmo_path.is_file():
        try:
            cosmo_payload = _load_json(cosmo_path)
            summary["cosmo"] = {
                "t_s": cosmo_payload.get("time_s"),
                "g_star": cosmo_payload.get("g_star"),
                "rho_J_m3": cosmo_payload.get("rho_J_m3"),
                "E_total_J": cosmo_payload.get("E_total_J"),
                "horizon_radius_m": cosmo_payload.get("horizon_radius_m"),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            summary["cosmo"] = {"error": "failed to load early_universe_t1s.json"}

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plots: List[Path] = []
    if enable_plots:
        noise_plot = _maybe_plot_noise(prefix.with_suffix(".noise.png"), list(noise_up.values()), list(noise_down.values()))
        if noise_plot:
            plots.append(noise_plot)
        frontier_plot = _maybe_plot_frontier(prefix.with_suffix(".frontier.png"), curve, {
            "kneedle": summary["frontier"]["knee_kneedle"],
            "second_derivative": summary["frontier"]["knee_second_deriv"],
        })
        if frontier_plot:
            plots.append(frontier_plot)
        drift_plot = _maybe_plot_drift(prefix.with_suffix(".drift.png"), drift_curve, alpha)
        if drift_plot:
            plots.append(drift_plot)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(summary_path, arcname=summary_path.name)
        for plot in plots:
            archive.write(plot, arcname=plot.name)

    return summary


if __name__ == "__main__":
    main()
