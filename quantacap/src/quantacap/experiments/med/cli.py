"""Command-line entry point for synthetic medicinal discovery (research only)."""

from __future__ import annotations

import argparse
import json
from quantacap.core.adapter_store import load_adapter

from .docking import run_search
from .report import build_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic docking demo (simulation)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    search = sub.add_parser("search", help="Run a Monte-Carlo binding search")
    search.add_argument("--target", required=True)
    search.add_argument("--cycles", type=int, default=5000)
    search.add_argument("--topk", type=int, default=10)
    search.add_argument("--seed", type=int, default=424242)
    search.add_argument("--id", dest="adapter_id", default=None)

    replay = sub.add_parser("replay", help="Replay previously stored candidates")
    replay.add_argument("--id", required=True)

    report = sub.add_parser("report", help="Flatten metrics into a JSON artifact")
    report.add_argument("--id", required=True)
    report.add_argument("--out", required=False)

    args = parser.parse_args(argv)

    if args.cmd == "search":
        payload = run_search(
            args.target,
            cycles=args.cycles,
            topk=args.topk,
            seed=args.seed,
            adapter_id=args.adapter_id,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.cmd == "replay":
        record = load_adapter(args.id)
        print(json.dumps(record, indent=2))
        return

    if args.cmd == "report":
        out = build_report(args.id, output=args.out)
        print(json.dumps(out, indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
