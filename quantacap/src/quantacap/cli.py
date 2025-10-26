from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict

import numpy as np

from quantacap.core.adapter_store import create_adapter, list_adapters, load_adapter
from quantacap.experiments.atom1d import run_atom1d
from quantacap.experiments.chsh import run_chsh
from quantacap.experiments.chsh_rehearsal import run_chsh_scan
from quantacap.quantum.backend import create_circuit
from quantacap.quantum.bell import bell_counts
from quantacap.quantum.gates import H, RZ
from quantacap.quantum.grover import grover_search
from quantacap.quantum.xp import to_numpy


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

    estimate = sub.add_parser("estimate", help="Estimate memory required for a state vector")
    estimate.add_argument("--n", type=int, required=True)
    estimate.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")

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

    atom = sub.add_parser("atom1d", help="Generate synthetic atom 1-D density")
    atom.add_argument("--n", type=int, required=True)
    atom.add_argument("--L", type=float, required=True)
    atom.add_argument("--sigma", type=float, required=True)
    atom.add_argument("--id", required=True)
    _add_backend_flags(atom, allow_backend=False)

    args = ap.parse_args()

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

    raise SystemExit(2)


if __name__ == "__main__":
    main()
