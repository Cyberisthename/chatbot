import argparse
import json

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

    args = ap.parse_args()

    if args.cmd == "grover":
        out = grover_search(n=args.n, marked_index=args.marked, shots=args.shots, iters=args.iters)
        print(json.dumps(out, indent=2))
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
