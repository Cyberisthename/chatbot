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
    module = optional_import(
        "quantacap.scripts.phase_transition_report",
        purpose="generate phase transition summary",
    )
    main = getattr(module, "main")
    kwargs: Dict[str, Any] = {}
    if getattr(args, "artifacts_dir", None) is not None:
        kwargs["artifacts_dir"] = args.artifacts_dir
    if getattr(args, "out_prefix", None) is not None:
        kwargs["output_prefix"] = args.out_prefix
    main(**kwargs)

import numpy as np

from quantacap.core.adapter_store import create_adapter, list_adapters, load_adapter
from quantacap.experiments.atom1d import run_atom1d
from quantacap.experiments.chsh import run_chsh
from quantacap.experiments.chsh_rehearsal import run_chsh_scan
from quantacap.experiments.chsh_ybit import run_chsh_y
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
