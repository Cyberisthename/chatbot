from __future__ import annotations

import argparse
import json
import math
import os
import runpy
from pathlib import Path
from typing import Any, Dict



def _report_discovery_cmd(_args: argparse.Namespace) -> None:
    runpy.run_module("scripts.collect_discovery", run_name="__main__")


def _zip_artifacts_cmd(_args: argparse.Namespace) -> None:
    runpy.run_module("scripts.zip_artifacts", run_name="__main__")


def _serve_download_cmd(args: argparse.Namespace) -> None:
    port = getattr(args, "port", 8009)
    os.environ["QUANTACAP_PORT"] = str(port)
    runpy.run_module("scripts.zip_artifacts", run_name="__main__")
    zip_path = os.path.join("artifacts", "quion_experiment.zip")
    print(
        "Starting download server...\n"
        f"  ZIP path: {zip_path}\n"
        f"  Local URL: http://localhost:{port}/download\n"
        "Press Ctrl+C to stop."
    )
    runpy.run_module("scripts.serve_download", run_name="__main__")


def _summarize_artifacts_cmd(args: argparse.Namespace) -> None:
    zip_path = Path(getattr(args, 'zip_path', os.path.join('artifacts', 'quion_experiment.zip')))
    artifacts_dir = Path(getattr(args, 'artifacts_dir', 'artifacts'))
    summary_path = Path(getattr(args, 'summary_path', os.path.join('artifacts', 'summary_results.json')))
    summarize_quantum_artifacts(zip_path, artifacts_dir, summary_path)


def _report_phase_transition_cmd(args: argparse.Namespace) -> None:
    """Generate the phase-transition bundle via the standalone script."""

    module = optional_import(
        "scripts.phase_transition_report",
        purpose="generate phase transition summary",
    )
    main = getattr(module, "main")
    kwargs: Dict[str, Any] = {}
    if getattr(args, "artifacts_dir", None) is not None:
        kwargs["artifacts_dir"] = args.artifacts_dir
    if getattr(args, "out_prefix", None) is not None:
        kwargs["output_prefix"] = args.out_prefix
    main(**kwargs)


def _maybe_plot_cosmo(out_path: Path, payload: Dict[str, Any]) -> Path | None:
    try:
        mpl = optional_import(
            "matplotlib",
            pip_name="matplotlib",
            purpose="visualise early-universe energy",
        )
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return

    density_values = [
        payload.get("rho_J_m3", float("nan")),
        payload.get("rho_modern_scaled_J_m3", float("nan")),
    ]
    density_labels = ["ρ(t)", "ρ(modern→t)"]

    energy_values = [payload.get("E_total_J", float("nan"))]
    energy_labels = ["E_total"]

    fig, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(9, 4))

    ax_d.bar(density_labels, density_values, color=["tab:blue", "tab:orange"])
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Energy density (J m⁻³)")
    ax_d.set_title("Radiation energy density")

    ax_e.bar(energy_labels, energy_values, color="tab:green")
    ax_e.set_yscale("log")
    ax_e.set_ylabel("Energy (J)")
    ax_e.set_title("Total energy in horizon")

    fig.suptitle("Early universe energy budget at t≈1 s")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_json(path: Path | str, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _maybe_plot_timecrystal(result, prefix: str) -> list[Path]:
    try:
        mpl = optional_import(
            "matplotlib",
            pip_name="matplotlib",
            purpose="plot time-crystal observables",
        )
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return []

    times = list(range(len(result.autocorrelation)))
    freq = result.frequencies
    spec = result.spectrum

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, result.autocorrelation, label="C(t)")
    ax.set_xlabel("Floquet step")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.set_title("Time-crystal autocorrelation")
    fig.tight_layout()
    ac_path = Path(f"{prefix}_autocorr.png")
    ac_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ac_path, dpi=150)
    plt.close(fig)
    paths.append(ac_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq, spec, color="tab:orange")
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("|C(ω)|")
    ax.set_title("Stroboscopic spectrum")
    ax.axvline(0.5, color="tab:red", linestyle="--", alpha=0.6)
    fig.tight_layout()
    sp_path = Path(f"{prefix}_spectrum.png")
    sp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sp_path, dpi=150)
    plt.close(fig)
    paths.append(sp_path)
    return paths


def _maybe_plot_material_time(result, prefix: str) -> list[Path]:
    try:
        mpl = optional_import(
            "matplotlib",
            pip_name="matplotlib",
            purpose="plot material-time dynamics",
        )
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return []

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.real_time, result.material_time, label="material time")
    ax.set_xlabel("Real time step")
    ax.set_ylabel("Material time ticks")
    ax.set_title("Material-time growth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    mt_path = Path(f"{prefix}_material_time.png")
    mt_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(mt_path, dpi=150)
    plt.close(fig)
    paths.append(mt_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.real_time, result.entropy_proxy, color="tab:green")
    ax.set_xlabel("Real time step")
    ax.set_ylabel("Entropy proxy")
    ax.set_title("Entropy evolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ent_path = Path(f"{prefix}_entropy.png")
    ent_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ent_path, dpi=150)
    plt.close(fig)
    paths.append(ent_path)
    return paths


def _maybe_plot_timerev(result, prefix: str) -> list[Path]:
    try:
        mpl = optional_import(
            "matplotlib",
            pip_name="matplotlib",
            purpose="plot quantum reversal fidelity",
        )
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.noise_levels, result.fidelities, marker="o")
    ax.set_xlabel("Noise strength")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Noisy reversal fidelity")
    fig.tight_layout()
    out_path = Path(f"{prefix}_fidelity.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


import numpy as np

from quantacap.core.adapter_store import create_adapter, list_adapters, load_adapter
from quantacap.cosmo.t1s import energy_at_1s
from quantacap.experiments.atom1d import run_atom1d
from quantacap.experiments.chsh import run_chsh
from quantacap.experiments.chsh_rehearsal import run_chsh_scan
from quantacap.experiments.chsh_ybit import run_chsh_y
from quantacap.experiments.exotic_atom_floquet import run_exotic_atom_floquet
from quantacap.experiments.pi_phase import run_pi_phase
from quantacap.quantum.backend import create_circuit
from quantacap.quantum.bell import bell_counts
from quantacap.quantum.gates import H, RZ
from quantacap.quantum.grover import grover_search
from quantacap.quantum.xp import to_numpy
from quantacap.primitives.ggraph import GGraph
from quantacap.primitives.ybit import YBit
from quantacap.primitives.zbit import ZBit
from quantacap.utils.optional_import import optional_import
from scripts.summarize_quantum_artifacts import summarize_quantum_artifacts


def _forward_cli(module: str, purpose: str, argv: list[str] | None) -> None:
    module_obj = optional_import(module, purpose=purpose)
    main = getattr(module_obj, 'main')
    main(argv or None)


def _virtual_element_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="virtual element search")

    from quantacap.experiments.virtual_element.plot import (
        plot_binding_curve,
        plot_heatmap,
        plot_top_candidates,
    )
    from quantacap.experiments.virtual_element.search import search_isotopes

    result = search_isotopes(
        Z_min=args.Z_min,
        Z_max=args.Z_max,
        A_min_offset=args.A_min_offset,
        A_max_offset=args.A_max_offset,
        n_mc=args.mc,
        seed=args.seed,
        adapter_top_k=args.topk,
    )
    payload: Dict[str, Any] = {
        "params": {
            "Z_min": args.Z_min,
            "Z_max": args.Z_max,
            "A_min_offset": args.A_min_offset,
            "A_max_offset": args.A_max_offset,
            "n_mc": args.mc,
            "seed": args.seed,
            "topk": args.topk,
        },
        "top": result["top"][: args.topk],
        "grid": {
            "Z_values": result["Z_values"],
            "A_values": result["A_values"],
        },
        "latency_ms": result["latency_ms"],
    }
    _write_json(args.out, payload)

    plot_paths: list[str] = []
    if args.plot:
        prefix = str(args.plot_prefix or Path(args.out).with_suffix(""))
        heatmap_path = plot_heatmap(
            result["Z_values"],
            result["A_values"],
            result["stability_matrix"],
            out_path=f"{prefix}_heatmap.png",
        )
        if heatmap_path:
            plot_paths.append(str(heatmap_path))
        top_plot = plot_top_candidates(result["entries"], out_path=f"{prefix}_top.png", top_k=args.topk)
        if top_plot:
            plot_paths.append(str(top_plot))
        top_Z = []
        for row in result["top"]:
            z_val = row["Z"]
            if z_val not in top_Z:
                top_Z.append(z_val)
            if len(top_Z) >= min(3, args.topk):
                break
        for z_val in top_Z:
            binding_path = plot_binding_curve(
                result["entries"],
                Z=z_val,
                out_path=f"{prefix}_binding_Z{z_val}.png",
            )
            if binding_path:
                plot_paths.append(str(binding_path))

    print(
        json.dumps(
            {
                "top": payload["top"],
                "out": str(args.out),
                "plots": plot_paths,
            },
            indent=2,
        )
    )


def _timecrystal_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="time-crystal simulation")
    from dataclasses import asdict

    from quantacap.experiments.timecrystal import run_time_crystal

    result = run_time_crystal(
        N=args.N,
        steps=args.steps,
        disorder=args.disorder,
        jitter=args.jitter,
        seed=args.seed,
    )
    payload = asdict(result)
    _write_json(args.out, payload)
    plots = []
    if args.plot:
        plots = [str(path) for path in _maybe_plot_timecrystal(result, args.plot_prefix)]
    print(json.dumps({"detected": result.detected, "plots": plots, "out": str(args.out)}, indent=2))


def _material_time_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="material-time simulation")
    from dataclasses import asdict

    from quantacap.experiments.material_time import simulate_material_time

    result = simulate_material_time(
        traps=args.traps,
        steps=args.steps,
        rate=args.rate,
        temperature=args.temperature,
        seed=args.seed,
        reversal_period=args.reversal_period,
    )
    payload = asdict(result)
    _write_json(args.out, payload)
    plots = []
    if args.plot:
        plots = [str(path) for path in _maybe_plot_material_time(result, args.plot_prefix)]
    print(
        json.dumps(
            {
                "reversal_success": result.reversal_success,
                "reversal_attempts": result.reversal_attempts,
                "plots": plots,
                "out": str(args.out),
            },
            indent=2,
        )
    )


def _timerev_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="quantum reversal sweep")
    from dataclasses import asdict

    from quantacap.experiments.timerev import run_quantum_reversal

    noise = args.noise or [0.0, 0.02, 0.05, 0.1]
    result = run_quantum_reversal(
        n_qubits=args.qubits,
        depth=args.depth,
        noise_levels=noise,
        seed=args.seed,
        guard_threshold=args.guard_threshold,
    )
    payload = asdict(result)
    _write_json(args.out, payload)
    plots = []
    if args.plot:
        plots = [str(path) for path in _maybe_plot_timerev(result, args.plot_prefix)]
    print(
        json.dumps(
            {
                "guard_triggered": result.guard_triggered,
                "fidelities": result.fidelities,
                "out": str(args.out),
                "plots": plots,
            },
            indent=2,
        )
    )


def _time_suite_cmd(args: argparse.Namespace) -> None:
    selections = [part.strip() for part in args.sub.split(",") if part.strip()]
    if not selections:
        selections = ["timecrystal", "material_time", "timerev"]

    from dataclasses import asdict

    summary: Dict[str, Any] = {}
    out_prefix = args.out_prefix

    if "timecrystal" in selections:
        from quantacap.experiments.timecrystal import run_time_crystal

        tc = run_time_crystal(seed=args.seed)
        tc_path = _write_json(f"{out_prefix}_timecrystal.json", asdict(tc))
        plots = _maybe_plot_timecrystal(tc, f"{out_prefix}_timecrystal") if args.plot else []
        summary["timecrystal"] = {
            "detected": tc.detected,
            "out": str(tc_path),
            "plots": [str(p) for p in plots],
        }

    if "material_time" in selections:
        from quantacap.experiments.material_time import simulate_material_time

        mt = simulate_material_time(seed=args.seed)
        mt_path = _write_json(f"{out_prefix}_material_time.json", asdict(mt))
        plots = _maybe_plot_material_time(mt, f"{out_prefix}_material_time") if args.plot else []
        summary["material_time"] = {
            "reversal_success": mt.reversal_success,
            "out": str(mt_path),
            "plots": [str(p) for p in plots],
        }

    if "timerev" in selections:
        from quantacap.experiments.timerev import run_quantum_reversal

        qr = run_quantum_reversal(seed=args.seed)
        qr_path = _write_json(f"{out_prefix}_timerev.json", asdict(qr))
        plots = _maybe_plot_timerev(qr, f"{out_prefix}_timerev") if args.plot else []
        summary["timerev"] = {
            "guard_triggered": qr.guard_triggered,
            "out": str(qr_path),
            "plots": [str(p) for p in plots],
        }

    print(json.dumps(summary, indent=2))


def _quantum_tunneling_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="quantum tunneling simulation")
    
    from quantacap.experiments.quantum_tunneling import run_quantum_tunneling
    
    plot_path = f"{args.plot_prefix}.png" if args.plot else None
    
    result = run_quantum_tunneling(
        energy=args.energy,
        barrier=args.barrier,
        steps=args.steps,
        n=args.n,
        barrier_width=args.barrier_width,
        dt=args.dt,
        seed=args.seed,
        out=args.out,
        plot=plot_path,
    )
    
    print(json.dumps({
        "final_transmission": result["final_transmission"],
        "final_reflection": result["final_reflection"],
        "out": result["artifacts"]["json"],
        "plot": result["artifacts"].get("plot"),
    }, indent=2))


def _exotic_atom_cmd(args: argparse.Namespace) -> None:
    optional_import("numpy", pip_name="numpy", purpose="exotic atom Floquet experiment")
    
    result = run_exotic_atom_floquet(
        N=args.N,
        steps=args.steps,
        dt=args.dt,
        drive_amp=args.drive_amp,
        drive_freq=args.drive_freq,
        J_nn=args.J_nn,
        J_lr=args.J_lr,
        alpha=args.alpha,
        seed=args.seed,
        make_gif=not args.no_gif,
        out_json=args.out,
    )
    
    print(json.dumps({
        "experiment": result["experiment"],
        "out": args.out,
        "artifacts": result["results"]["artifacts"],
    }, indent=2))


def _add_backend_flags(parser: argparse.ArgumentParser, *, allow_backend: bool = True) -> None:
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0, help="Use GPU backend if available")
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    if allow_backend:
        parser.add_argument("--backend", choices=("statevector", "mps"), default="statevector")
        parser.add_argument("--chi", type=int, default=32, help="MPS bond dimension (when backend=mps)")


def _backend_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": getattr(args, "backend", "statevector"),
        "use_gpu": bool(getattr(args, "gpu", 0)),
        "dtype": getattr(args, "dtype", "complex128"),
        "chi": getattr(args, "chi", 32),
    }


def _backend_info(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "type": getattr(args, "backend", "statevector"),
        "device": "gpu" if getattr(args, "gpu", 0) else "cpu",
        "chi": getattr(args, "chi", None) if getattr(args, "backend", "statevector") == "mps" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantacap CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    gro = sub.add_parser("grover", help="Run Grover search (n=3 default)")
    gro.add_argument("--n", type=int, default=3)
    gro.add_argument("--marked", type=int, default=5)
    gro.add_argument("--shots", type=int, default=4096)
    gro.add_argument("--iters", type=int, default=None)
    _add_backend_flags(gro, allow_backend=False)

    fringe = sub.add_parser("fringe", help="Phase fringe scan with H-RZ-H")
    fringe.add_argument("--steps", type=int, default=73)
    fringe.add_argument("--shots", type=int, default=2048)
    fringe.add_argument("--seed", type=int, default=424242)
    _add_backend_flags(fringe)

    bell = sub.add_parser("bell", help="Run Bell-state experiment")
    bell.add_argument("--shots", type=int, default=8192)
    bell.add_argument("--seed", type=int, default=424242)
    bell.add_argument("--depol", type=float, default=0.0)
    bell.add_argument("--phase", type=float, default=0.0)
    _add_backend_flags(bell)

    save_state = sub.add_parser("save-state", help="Persist a prepared state to the adapter store")
    save_state.add_argument("--id", required=True)
    save_state.add_argument("--theta", type=float, required=True)
    save_state.add_argument("--n", type=int, default=1)
    save_state.add_argument("--seed", type=int, default=424242)
    _add_backend_flags(save_state)

    replay = sub.add_parser("replay", help="Replay a saved adapter (no recompute)")
    replay.add_argument("--id", required=True)

    ls_adapters = sub.add_parser("ls-adapters", help="List saved adapters")
    ls_adapters.add_argument("--prefix", default=None)

    report = sub.add_parser(
        "report-discovery", help="Generate a unified discovery report (JSON/MD/ZIP)"
    )
    report.set_defaults(func=_report_discovery_cmd)

    phase_report = sub.add_parser(
        "report-phase-transition",
        help="Analyse π-phase noise/coupling/drift phase transitions",
    )
    phase_report.add_argument("--artifacts-dir", default="artifacts")
    phase_report.add_argument("--out-prefix", default=None)
    phase_report.set_defaults(func=_report_phase_transition_cmd)

    bigbang = sub.add_parser(
        "bigbang-t1s",
        help="Estimate radiation energy at t≈1 s after the Big Bang",
    )
    bigbang.add_argument("--t", type=float, default=1.0)
    bigbang.add_argument("--g-star", dest="g_star", type=float, default=10.75)
    bigbang.add_argument("--T-MeV", dest="T_MeV", type=float, default=1.0)
    bigbang.add_argument(
        "--out",
        default=os.path.join("artifacts", "early_universe_t1s.json"),
        help="Output JSON path",
    )
    bigbang.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate a companion PNG when matplotlib is available",
    )

    virtual = sub.add_parser(
        "virtual-element",
        help="Run the virtual superheavy isotope search",
    )
    virtual.add_argument("--Zmin", dest="Z_min", type=int, default=100)
    virtual.add_argument("--Zmax", dest="Z_max", type=int, default=130)
    virtual.add_argument("--Amin-offset", dest="A_min_offset", type=int, default=150)
    virtual.add_argument("--Amax-offset", dest="A_max_offset", type=int, default=220)
    virtual.add_argument("--mc", type=int, default=5, help="Monte-Carlo samples per isotope")
    virtual.add_argument("--seed", type=int, default=424242)
    virtual.add_argument("--topk", type=int, default=10)
    virtual.add_argument(
        "--out",
        default=os.path.join("artifacts", "virtual_elements.json"),
        help="Output JSON path",
    )
    virtual.add_argument(
        "--plot-prefix",
        default=os.path.join("artifacts", "virtual_elements"),
        help="Prefix for generated plots",
    )
    virtual.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate heatmap and ranking plots when matplotlib is available",
    )
    virtual.set_defaults(func=_virtual_element_cmd)

    timecrystal_parser = sub.add_parser(
        "timecrystal", help="Simulate a discrete time-crystal Floquet drive"
    )
    timecrystal_parser.add_argument("--N", type=int, default=10)
    timecrystal_parser.add_argument("--steps", type=int, default=500)
    timecrystal_parser.add_argument("--disorder", type=float, default=0.08)
    timecrystal_parser.add_argument("--jitter", type=float, default=0.05)
    timecrystal_parser.add_argument("--seed", type=int, default=424242)
    timecrystal_parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "timecrystal.json"),
    )
    timecrystal_parser.add_argument(
        "--plot-prefix",
        default=os.path.join("artifacts", "timecrystal"),
    )
    timecrystal_parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate autocorrelation and spectrum plots",
    )
    timecrystal_parser.set_defaults(func=_timecrystal_cmd)

    material_parser = sub.add_parser(
        "material-time", help="Run the material-time aging simulation"
    )
    material_parser.add_argument("--traps", type=int, default=64)
    material_parser.add_argument("--steps", type=int, default=5000)
    material_parser.add_argument("--rate", type=float, default=1e-3)
    material_parser.add_argument("--temperature", type=float, default=0.3)
    material_parser.add_argument("--reversal-period", dest="reversal_period", type=int, default=250)
    material_parser.add_argument("--seed", type=int, default=424242)
    material_parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "material_time.json"),
    )
    material_parser.add_argument(
        "--plot-prefix",
        default=os.path.join("artifacts", "material_time"),
    )
    material_parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate material-time plots when matplotlib is available",
    )
    material_parser.set_defaults(func=_material_time_cmd)

    timerev_parser = sub.add_parser(
        "timerev", help="Sweep noisy quantum reversal fidelity"
    )
    timerev_parser.add_argument("--qubits", type=int, default=3)
    timerev_parser.add_argument("--depth", type=int, default=12)
    timerev_parser.add_argument(
        "--noise",
        nargs="*",
        type=float,
        help="Explicit noise schedule (space-separated values)",
    )
    timerev_parser.add_argument("--guard-threshold", type=float, default=0.05)
    timerev_parser.add_argument("--seed", type=int, default=424242)
    timerev_parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "timerev.json"),
    )
    timerev_parser.add_argument(
        "--plot-prefix",
        default=os.path.join("artifacts", "timerev"),
    )
    timerev_parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate fidelity plot when matplotlib is available",
    )
    timerev_parser.set_defaults(func=_timerev_cmd)

    suite_parser = sub.add_parser(
        "time-suite", help="Run a bundle of time-dynamics experiments"
    )
    suite_parser.add_argument(
        "--sub",
        default="timecrystal,material_time,timerev",
        help="Comma-separated list of experiments to execute",
    )
    suite_parser.add_argument("--seed", type=int, default=424242)
    suite_parser.add_argument(
        "--out-prefix",
        default=os.path.join("artifacts", "time_suite"),
        help="Prefix for generated JSON/plots",
    )
    suite_parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate plots for the selected experiments",
    )
    suite_parser.set_defaults(func=_time_suite_cmd)

    tunneling_parser = sub.add_parser(
        "quantum-tunneling", help="Simulate 1D quantum tunneling through a potential barrier"
    )
    tunneling_parser.add_argument("--energy", type=float, default=2.0, help="Particle kinetic energy")
    tunneling_parser.add_argument("--barrier", type=float, default=5.0, help="Potential barrier height")
    tunneling_parser.add_argument("--steps", type=int, default=2000, help="Number of time evolution steps")
    tunneling_parser.add_argument("--n", type=int, default=1024, help="Number of spatial grid points")
    tunneling_parser.add_argument("--barrier-width", dest="barrier_width", type=int, default=128, help="Barrier width in grid points")
    tunneling_parser.add_argument("--dt", type=float, default=0.002, help="Time step size")
    tunneling_parser.add_argument("--seed", type=int, default=424242)
    tunneling_parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "tunneling_result.json"),
        help="Output JSON path",
    )
    tunneling_parser.add_argument(
        "--plot-prefix",
        default=os.path.join("artifacts", "tunneling_evolution"),
        help="Prefix for generated plot",
    )
    tunneling_parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate evolution plot when matplotlib is available",
    )
    tunneling_parser.set_defaults(func=_quantum_tunneling_cmd)

    exotic_atom_parser = sub.add_parser(
        "exotic-atom", help="Run exotic atom Floquet experiment with long-range Hamiltonian"
    )
    exotic_atom_parser.add_argument("--N", type=int, default=8, help="Number of qubits")
    exotic_atom_parser.add_argument("--steps", type=int, default=80, help="Number of evolution steps")
    exotic_atom_parser.add_argument("--dt", type=float, default=0.05, help="Time step size")
    exotic_atom_parser.add_argument("--drive-amp", dest="drive_amp", type=float, default=1.0, help="Drive amplitude")
    exotic_atom_parser.add_argument("--drive-freq", dest="drive_freq", type=float, default=2.0, help="Drive frequency")
    exotic_atom_parser.add_argument("--J-nn", dest="J_nn", type=float, default=1.0, help="Nearest-neighbor coupling")
    exotic_atom_parser.add_argument("--J-lr", dest="J_lr", type=float, default=0.5, help="Long-range coupling strength")
    exotic_atom_parser.add_argument("--alpha", type=float, default=1.5, help="Long-range decay exponent")
    exotic_atom_parser.add_argument("--seed", type=int, default=424242)
    exotic_atom_parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    exotic_atom_parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "exotic_atom_floquet.json"),
        help="Output JSON path",
    )
    exotic_atom_parser.set_defaults(func=_exotic_atom_cmd)

    zip_parser = sub.add_parser(
        "zip-artifacts", help="Zip artifacts into artifacts/quion_experiment.zip"
    )
    zip_parser.set_defaults(func=_zip_artifacts_cmd)

    serve_parser = sub.add_parser(
        "serve-download", help="Serve the artifacts ZIP over HTTP"
    )
    serve_parser.add_argument("--port", type=int, default=8009)
    serve_parser.set_defaults(func=_serve_download_cmd)

    med_parser = sub.add_parser('med', help='Forward to medicinal discovery CLI')
    med_parser.add_argument('argv', nargs=argparse.REMAINDER)
    med_parser.set_defaults(func=lambda args: _forward_cli('quantacap.experiments.med.cli', 'run medicinal discovery', args.argv))

    viz3d_parser = sub.add_parser('viz3d', help='Forward to 3D computing map CLI')
    viz3d_parser.add_argument('argv', nargs=argparse.REMAINDER)
    viz3d_parser.set_defaults(func=lambda args: _forward_cli('quantacap.viz3d.cli', 'render 3D computing maps', args.argv))

    astro_parser = sub.add_parser('astro', help='Forward to Schwarzschild simulator CLI')
    astro_parser.add_argument('argv', nargs=argparse.REMAINDER)
    astro_parser.set_defaults(func=lambda args: _forward_cli('quantacap.astro.cli', 'run black-hole simulations', args.argv))

    pi_parser = sub.add_parser('pi-ext', help='Forward to extended π experiments CLI')
    pi_parser.add_argument('argv', nargs=argparse.REMAINDER)
    pi_parser.set_defaults(func=lambda args: _forward_cli('quantacap.experiments.pi.cli', 'run π-phase extension experiments', args.argv))

    summarize_parser = sub.add_parser("summarize-artifacts", help="Summarize saved quantum artifacts")
    summarize_parser.add_argument("--zip", dest="zip_path", default=os.path.join("artifacts", "quion_experiment.zip"))
    summarize_parser.add_argument("--artifacts", dest="artifacts_dir", default="artifacts")
    summarize_parser.add_argument("--output", dest="summary_path", default=os.path.join("artifacts", "summary_results.json"))
    summarize_parser.set_defaults(func=_summarize_artifacts_cmd)

    estimate = sub.add_parser("estimate", help="Estimate memory required for a state vector")
    estimate.add_argument("--n", type=int, required=True)
    estimate.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")

    pi_phase = sub.add_parser("pi-phase", help="Run probabilistic pi phase rotations")
    pi_phase.add_argument("--rotations", type=int, default=100_000)
    pi_phase.add_argument("--precision", type=float, default=1e-9)
    pi_phase.add_argument("--seed", type=int, default=424242)
    pi_phase.add_argument("--samples", type=int, default=256)
    pi_phase.add_argument("--id", default=None)
    pi_phase.add_argument("--artifact", default=None)

    chsh = sub.add_parser("chsh", help="Run CHSH Bell-inequality experiment")
    chsh.add_argument("--shots", type=int, default=50000)
    chsh.add_argument("--depol", type=float, default=0.0)
    chsh.add_argument("--seed", type=int, default=424242)
    _add_backend_flags(chsh)

    chsh_scan = sub.add_parser("chsh-scan", help="Run CHSH noise rehearsal scan")
    chsh_scan.add_argument("--pmin", type=float, default=0.0)
    chsh_scan.add_argument("--pmax", type=float, default=0.2)
    chsh_scan.add_argument("--steps", type=int, default=21)
    chsh_scan.add_argument("--shots", type=int, default=50000)
    chsh_scan.add_argument("--id", required=True)
    chsh_scan.add_argument("--seed", type=int, default=424242)
    _add_backend_flags(chsh_scan)

    chsh_y = sub.add_parser("chsh-y", help="Run CHSH with Y-bit and G-graph modulation")
    chsh_y.add_argument("--shots", type=int, default=50000)
    chsh_y.add_argument("--depol", type=float, default=0.0)
    chsh_y.add_argument("--seed", type=int, default=424242)
    chsh_y.add_argument("--seed-id", default="demo.ybit")
    chsh_y.add_argument("--lam", type=float, default=0.85)
    chsh_y.add_argument("--eps", type=float, default=0.02)
    chsh_y.add_argument("--delta", type=float, default=0.03)
    chsh_y.add_argument("--nodes", type=int, default=4096)
    chsh_y.add_argument("--out", type=int, default=3)
    chsh_y.add_argument("--gamma", type=float, default=0.87)
    chsh_y.add_argument("--id", default=None)
    _add_backend_flags(chsh_y)

    build_g = sub.add_parser("build-g", help="Build and persist a G-graph summary")
    build_g.add_argument("--nodes", type=int, default=4096)
    build_g.add_argument("--out", type=int, default=3)
    build_g.add_argument("--gamma", type=float, default=0.87)
    build_g.add_argument("--seed", type=int, default=424242)
    build_g.add_argument("--id", default=None)

    make_y = sub.add_parser("make-y", help="Synthesize a demo Y-bit from a single qubit state")
    make_y.add_argument("--theta", type=float, default=1.234)
    make_y.add_argument("--lam", type=float, default=0.85)
    make_y.add_argument("--eps", type=float, default=0.02)
    make_y.add_argument("--seed", type=int, default=424242)
    _add_backend_flags(make_y, allow_backend=False)

    atom = sub.add_parser("atom1d", help="Generate synthetic atom 1-D density")
    atom.add_argument("--n", type=int, required=True)
    atom.add_argument("--L", type=float, required=True)
    atom.add_argument("--sigma", type=float, required=True)
    atom.add_argument("--id", required=True)
    _add_backend_flags(atom, allow_backend=False)

    quion_viz = sub.add_parser("quion-viz", help="Run Quion++ visualisation scenarios")
    quion_viz.add_argument("--scenario", choices=("reverse", "freeze", "noise", "chshy"), required=True)
    quion_viz.add_argument("--steps", type=int, default=1000)
    quion_viz.add_argument("--stride", type=int, default=10)
    quion_viz.add_argument("--tau", type=float, default=1e-4)
    quion_viz.add_argument("--jitter", type=float, default=1e-3)
    quion_viz.add_argument("--delta", type=float, default=0.03)
    quion_viz.add_argument("--lam", type=float, default=0.85)
    quion_viz.add_argument("--eps", type=float, default=0.02)
    quion_viz.add_argument("--seed", type=int, default=424242)
    quion_viz.add_argument("--seed-id", default="demo.ybit")
    quion_viz.add_argument("--video-format", choices=("auto", "mp4", "gif"), default="auto")
    quion_viz.add_argument("--out-prefix", default="artifacts/quion")
    quion_viz.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")

    args = ap.parse_args()

    if hasattr(args, "func"):
        args.func(args)
        return

    if args.cmd == "grover":
        out = grover_search(n=args.n, marked_index=args.marked, shots=args.shots, iters=args.iters)
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "fringe":
        steps = max(2, args.steps)
        thetas = np.linspace(0.0, 2 * math.pi, steps)
        backend_cfg = _backend_kwargs(args)
        p1 = []
        for theta in thetas:
            circuit = create_circuit(1, seed=args.seed, **backend_cfg)
            xp = getattr(circuit, "xp", None)
            circuit.add(H(xp=xp), [0])
            circuit.add(RZ(float(theta), xp=xp), [0])
            circuit.add(H(xp=xp), [0])
            prob = circuit.probs()[1]
            p1.append(float(prob))
        out = {"theta": thetas.tolist(), "p1": p1}
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "bell":
        backend_cfg = _backend_kwargs(args)
        noise: Dict[str, float] | None = {}
        if args.depol and args.depol > 0:
            noise["depol"] = float(args.depol)
        if args.phase and args.phase > 0:
            noise["phase"] = float(args.phase)
        if not noise:
            noise = None
        counts = bell_counts(shots=args.shots, seed=args.seed, noise=noise, **backend_cfg)
        out = {"shots": args.shots, "counts": counts, "noise": noise or {}, "backend": backend_cfg}
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "save-state":
        backend_cfg = _backend_kwargs(args)
        if args.n != 1:
            raise SystemExit("save-state currently supports n=1 only")
        circuit = create_circuit(args.n, seed=args.seed, **backend_cfg)
        xp = getattr(circuit, "xp", None)
        xp_module = xp if xp is not None else np
        circuit.add(H(xp=xp), [0])
        circuit.add(RZ(float(args.theta), xp=xp), [0])
        circuit.add(H(xp=xp), [0])
        psi = to_numpy(xp_module, circuit.run()).reshape(-1)
        probs = circuit.probs().tolist()
        backend_info = _backend_info(args)
        path = create_adapter(
            args.id,
            state={
                "n": args.n,
                "dtype": args.dtype,
                "amps": psi,
                "probs": probs,
                "backend": backend_info,
            },
            meta={"theta": float(args.theta)},
        )
        print(json.dumps({"saved": path}, indent=2))
        return

    if args.cmd == "replay":
        record = load_adapter(args.id)
        print(json.dumps(record, indent=2))
        return

    if args.cmd == "ls-adapters":
        print(json.dumps(list_adapters(args.prefix), indent=2))
        return

    if args.cmd == "estimate":
        n = args.n
        bytes_per_amp = 8 if args.dtype == "complex64" else 16
        total_bytes = bytes_per_amp * (1 << n)
        gib = total_bytes / (1024**3)
        print(json.dumps({"n": n, "dtype": args.dtype, "bytes": total_bytes, "GiB": gib}, indent=2))
        return

    if args.cmd == "bigbang-t1s":
        result = energy_at_1s(t=args.t, g_star=args.g_star, T_MeV=args.T_MeV)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        if getattr(args, 'plot', True):
            png_path = out_path.with_suffix('.png')
            created = _maybe_plot_cosmo(png_path, result)
            if created is not None:
                result['plot'] = str(created)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "pi-phase":
        result = run_pi_phase(
            rotations=args.rotations,
            precision=float(args.precision),
            seed=args.seed,
            samples=args.samples,
            adapter_id=args.id,
            artifact_path=args.artifact,
        )
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "chsh":
        backend_cfg = _backend_kwargs(args)
        result = run_chsh(
            shots=args.shots,
            depol=args.depol,
            seed=args.seed,
            **backend_cfg,
        )
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "chsh-scan":
        backend_cfg = _backend_kwargs(args)
        result = run_chsh_scan(
            pmin=args.pmin,
            pmax=args.pmax,
            steps=args.steps,
            shots=args.shots,
            adapter_id=args.id,
            seed=args.seed,
            **backend_cfg,
        )
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "chsh-y":
        backend_cfg = _backend_kwargs(args)
        adapter_id = args.id or f"chsh_y.{args.seed_id}"
        result = run_chsh_y(
            shots=args.shots,
            depol=args.depol,
            seed=args.seed,
            seed_id=args.seed_id,
            lam=args.lam,
            eps=args.eps,
            delta=args.delta,
            graph_nodes=args.nodes,
            graph_out=args.out,
            graph_gamma=args.gamma,
            adapter_id=adapter_id,
            **backend_cfg,
        )
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "build-g":
        graph = GGraph(n=args.nodes, out_degree=args.out, gamma=args.gamma, seed=args.seed)
        summary = graph.summary()
        adapter_id = args.id or f"ggraph.n{args.nodes}.gamma{args.gamma:.2f}"
        create_adapter(adapter_id, data=summary, meta={"kind": "ggraph"})
        print(json.dumps({"id": adapter_id, "summary": summary}, indent=2))
        return

    if args.cmd == "make-y":
        backend_cfg = _backend_kwargs(args)
        circuit = create_circuit(1, seed=args.seed, **backend_cfg)
        xp = getattr(circuit, "xp", None)
        circuit.add(H(xp=xp), [0])
        circuit.add(RZ(float(args.theta), xp=xp), [0])
        circuit.add(H(xp=xp), [0])
        psi = to_numpy(xp if xp is not None else np, circuit.run()).reshape(-1)
        alpha, beta = psi[0], psi[1]
        zbit = ZBit(seed=args.seed)
        ybit = YBit((alpha, beta), zbit, lam=args.lam, eps_phase=args.eps)
        out = {
            "z_value": zbit.value(),
            "bias": zbit.bias_sigma(),
            "phase": zbit.phase(),
            "ybit": ybit.info(),
        }
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "atom1d":
        result = run_atom1d(
            n=args.n,
            L=args.L,
            sigma=args.sigma,
            adapter_id=args.id,
            use_gpu=bool(args.gpu),
            dtype=args.dtype,
        )
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "quion-viz":
        viz_module = optional_import(
            "quantacap.experiments.quion_vizrun",
            pip_name="matplotlib",
            purpose="run Quion++ visualisation scenarios",
        )
        run_quion_viz = getattr(viz_module, "run_quion_viz")
        summary = run_quion_viz(
            scenario=args.scenario,
            steps=args.steps,
            stride=args.stride,
            tau=args.tau,
            jitter=args.jitter,
            delta=args.delta,
            lam=args.lam,
            eps=args.eps,
            dtype=args.dtype,
            seed=args.seed,
            seed_id=args.seed_id,
            out_prefix=args.out_prefix,
            video_format=args.video_format,
        )
        print(json.dumps(summary, indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
