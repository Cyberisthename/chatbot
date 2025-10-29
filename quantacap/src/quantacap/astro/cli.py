"""CLI entry point for synthetic Schwarzschild experiments."""

from __future__ import annotations

import argparse
import json

from .lensing import render_lensing_map
from .schwarzschild import integrate_null_geodesic


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic black-hole simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    lens = sub.add_parser("lens", help="Render an Einstein ring intensity map")
    lens.add_argument("--res", type=int, required=True)
    lens.add_argument("--impact-min", type=float, required=True)
    lens.add_argument("--impact-max", type=float, required=True)
    lens.add_argument("--id", default=None)

    geo = sub.add_parser("geodesic", help="Integrate a single null geodesic")
    geo.add_argument("--b", type=float, required=True)
    geo.add_argument("--steps", type=int, default=5000)

    args = parser.parse_args(argv)

    if args.cmd == "lens":
        meta = render_lensing_map(
            resolution=args.res,
            impact_min=args.impact_min,
            impact_max=args.impact_max,
            adapter_id=args.id,
        )
        print(json.dumps(meta, indent=2))
        return

    if args.cmd == "geodesic":
        result = integrate_null_geodesic(args.b, steps=args.steps)
        print(json.dumps(result.to_dict(), indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
