"""Command-line interface for running Quantacap experiments."""

import argparse
import json
import math

import numpy as np

from quantacap.core.adapter_store import create_adapter, list_adapters, load_adapter
from quantacap.quantum.bell import bell_counts
from quantacap.quantum.circuits import Circuit
from quantacap.quantum.gates import H, RZ
from quantacap.quantum.grover import grover_search


def main():
    ap = argparse.ArgumentParser(description="Quantacap CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Grover command
    gro = sub.add_parser("grover", help="Run Grover search (n=3 default)")
    gro.add_argument("--n", type=int, default=3)
    gro.add_argument("--marked", type=int, default=5)
    gro.add_argument("--shots", type=int, default=4096)
    gro.add_argument("--iters", type=int, default=None)

    fringe = sub.add_parser("fringe", help="Phase fringe scan with H-RZ-H")
    fringe.add_argument("--steps", type=int, default=73)
    fringe.add_argument("--shots", type=int, default=2048)
    fringe.add_argument("--seed", type=int, default=424242)

    bell = sub.add_parser("bell", help="Run Bell-state experiment")
    bell.add_argument("--shots", type=int, default=8192)
    bell.add_argument("--seed", type=int, default=424242)
    bell.add_argument("--depol", type=float, default=0.0)
    bell.add_argument("--phase", type=float, default=0.0)

    save_state = sub.add_parser("save-state", help="Persist a prepared state to the adapter store")
    save_state.add_argument("--id", required=True)
    save_state.add_argument("--theta", type=float, required=True)
    save_state.add_argument("--n", type=int, default=1)
    save_state.add_argument("--seed", type=int, default=424242)

    replay = sub.add_parser("replay", help="Replay a saved adapter (no recompute)")
    replay.add_argument("--id", required=True)

    ls_adapters = sub.add_parser("ls-adapters", help="List saved adapters")
    ls_adapters.add_argument("--prefix", default=None)

    args = ap.parse_args()

    if args.cmd == "grover":
        out = grover_search(n=args.n, marked_index=args.marked, shots=args.shots, iters=args.iters)
        print(json.dumps(out, indent=2))
    elif args.cmd == "fringe":
        steps = max(2, args.steps)
        thetas = np.linspace(0.0, 2 * math.pi, steps)
        p1 = []
        for theta in thetas:
            circuit = Circuit(n=1, seed=args.seed)
            circuit.add(H(), [0])
            circuit.add(RZ(float(theta)), [0])
            circuit.add(H(), [0])
            probs = circuit.probs()
            p1.append(float(probs[1]))
        out = {"theta": thetas.tolist(), "p1": p1}
        print(json.dumps(out, indent=2))
    elif args.cmd == "bell":
        noise = {}
        if args.depol and args.depol > 0:
            noise["depol"] = float(args.depol)
        if args.phase and args.phase > 0:
            noise["phase"] = float(args.phase)
        if not noise:
            noise = None
        counts = bell_counts(shots=args.shots, seed=args.seed, noise=noise)
        out = {"shots": args.shots, "counts": counts, "noise": noise or {}}
        print(json.dumps(out, indent=2))
    elif args.cmd == "save-state":
        if args.n != 1:
            raise SystemExit("save-state currently supports n=1 only")
        circuit = Circuit(n=args.n, seed=args.seed)
        circuit.add(H(), [0])
        circuit.add(RZ(float(args.theta)), [0])
        circuit.add(H(), [0])
        psi = circuit.run().reshape(-1)
        probs = circuit.probs().tolist()
        path = create_adapter(
            args.id,
            state={"n": args.n, "amps": psi, "probs": probs},
            meta={"theta": float(args.theta)},
        )
        print(json.dumps({"saved": path}, indent=2))
    elif args.cmd == "replay":
        record = load_adapter(args.id)
        print(json.dumps(record, indent=2))
    elif args.cmd == "ls-adapters":
        print(json.dumps(list_adapters(args.prefix), indent=2))
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
