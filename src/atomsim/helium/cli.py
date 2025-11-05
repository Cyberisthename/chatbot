"""CLI commands for helium atom simulations."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import solver


def he_ground(args: argparse.Namespace) -> None:
    """Run mean-field helium solver."""

    print(f"[HELIUM] N={args.N}, L={args.L}, steps={args.steps}, mix={args.mix}, spin={args.spin}")
    result = solver.solve_helium(
        N=args.N,
        L=args.L,
        steps=args.steps,
        dt=args.dt,
        eps=args.eps,
        mix=args.mix,
        spin=args.spin,
        seed=solver.SEED,
    )
    print(f"  → Total energy: {result.total_energy:.6f} hartree")
    print(f"  → Electron-electron: {result.ee_energy:.6f} hartree")

    out_dir = Path(args.out)
    solver.save_artifacts(result, out_dir, level=args.level)
    print(f"  → Artifacts saved to {out_dir}")


def register_helium_commands(subparsers) -> None:
    """Register helium CLI subcommands."""

    p_he = subparsers.add_parser("he-ground", help="Solve helium atom (mean-field)")
    p_he.add_argument("--N", type=int, default=256)
    p_he.add_argument("--L", type=float, default=12.0)
    p_he.add_argument("--steps", type=int, default=3000)
    p_he.add_argument("--dt", type=float, default=0.002)
    p_he.add_argument("--eps", type=float, default=0.3)
    p_he.add_argument("--mix", type=float, default=0.4, help="Density mixing parameter")
    p_he.add_argument("--spin", type=str, default="singlet", choices=["singlet", "triplet"])
    p_he.add_argument("--level", type=float, default=0.2, help="Isosurface level")
    p_he.add_argument("--out", type=str, required=True)
    p_he.set_defaults(func=he_ground)


__all__ = ["register_helium_commands"]
