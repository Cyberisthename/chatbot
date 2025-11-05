"""CLI commands for hydrogen atom simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from . import solver


def _parse_nlm(text: str) -> Tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--nlm requires three comma-separated integers")
    try:
        n, l, m = (int(p.strip()) for p in parts)
    except ValueError as exc:  # pragma: no cover - guarded by argparse
        raise argparse.ArgumentTypeError("--nlm values must be integers") from exc
    return n, l, m


def hyd_ground(args: argparse.Namespace) -> None:
    """Run ground-state hydrogen solver."""

    print(f"[HYDROGEN GROUND] N={args.N}, L={args.L}, Z={args.Z}, steps={args.steps}")
    result = solver.solve_ground(
        N=args.N,
        L=args.L,
        Z=args.Z,
        steps=args.steps,
        dt=args.dt,
        eps=args.eps,
        seed=solver.SEED,
    )
    print(f"  → Final energy: {result.energy:.6f} hartree")

    out_dir = Path(args.out)
    solver.save_artifacts(result, out_dir, analytic_label="1s", level=args.level, box_length=args.L)
    print(f"  → Artifacts saved to {out_dir}")


def hyd_excited(args: argparse.Namespace) -> None:
    """Run excited-state hydrogen solver."""

    n, l, m = args.nlm
    print(f"[HYDROGEN EXCITED] (n={n}, l={l}, m={m}) @ N={args.N}, steps={args.steps}")
    result = solver.solve_excited(
        n=n,
        l=l,
        m=m,
        N=args.N,
        L=args.L,
        Z=args.Z,
        steps=args.steps,
        dt=args.dt,
        eps=args.eps,
        seed=solver.SEED,
    )
    print(f"  → Final energy: {result.energy:.6f} hartree")

    out_dir = Path(args.out)
    solver.save_artifacts(result, out_dir, level=args.level, box_length=args.L)
    print(f"  → Artifacts saved to {out_dir}")


def register_hydrogen_commands(subparsers) -> None:
    """Register hydrogen CLI subcommands."""

    p_ground = subparsers.add_parser("hyd-ground", help="Solve hydrogen ground state")
    p_ground.add_argument("--N", type=int, default=256, help="Grid size per dimension")
    p_ground.add_argument("--L", type=float, default=12.0, help="Box side length (Bohr radii)")
    p_ground.add_argument("--Z", type=float, default=1.0, help="Nuclear charge")
    p_ground.add_argument("--steps", type=int, default=1200, help="Imaginary-time steps")
    p_ground.add_argument("--dt", type=float, default=0.002, help="Time step")
    p_ground.add_argument("--eps", type=float, default=0.3, help="Softening parameter")
    p_ground.add_argument("--level", type=float, default=0.2, help="Isosurface level")
    p_ground.add_argument("--out", type=str, required=True, help="Output directory")
    p_ground.set_defaults(func=hyd_ground)

    p_excited = subparsers.add_parser("hyd-excited", help="Solve hydrogen excited state")
    p_excited.add_argument("--N", type=int, default=256)
    p_excited.add_argument("--L", type=float, default=12.0)
    p_excited.add_argument("--Z", type=float, default=1.0)
    p_excited.add_argument("--steps", type=int, default=2000)
    p_excited.add_argument("--dt", type=float, default=0.002)
    p_excited.add_argument("--eps", type=float, default=0.3)
    p_excited.add_argument("--level", type=float, default=0.15, help="Isosurface level")
    p_excited.add_argument("--nlm", type=_parse_nlm, required=True, help="Quantum numbers n,l,m")
    p_excited.add_argument("--out", type=str, required=True)
    p_excited.set_defaults(func=hyd_excited)


__all__ = ["register_hydrogen_commands"]
