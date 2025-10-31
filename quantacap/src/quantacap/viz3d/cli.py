"""CLI entry for 3D computing map visualiser."""

from __future__ import annotations

import argparse
import json

from .fieldmap import build_field_series
from .scene import export_scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="3D computing map generator")
    parser.add_argument("--source", required=True)
    parser.add_argument("--field", choices=("amplitude", "phase", "entropy"), default="amplitude")
    parser.add_argument("--grid", nargs=3, type=int, metavar=("NX", "NY", "NZ"), default=(16, 16, 16))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    grid = tuple(args.grid)
    frames = build_field_series(source=args.source, field=args.field, grid=grid, steps=args.steps)
    meta = export_scene(frames, out_prefix=args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
