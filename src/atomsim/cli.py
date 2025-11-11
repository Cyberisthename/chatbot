"""Main CLI entry point for the atomsim 3D atom suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .fields import perturb
from .helium import cli as helium_cli
from .hydrogen import cli as hydrogen_cli
from .inverse import tomo


def hyd_field(args: argparse.Namespace) -> None:
    """Apply external field perturbation to a hydrogen state."""

    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"Input directory {in_dir} not found")
        sys.exit(1)

    psi = np.load(in_dir / "psi.npy")
    potential = np.load(in_dir / "potential.npy")
    density_ref = np.load(in_dir / "density.npy")

    out_dir = Path(args.out)

    grid_size = psi.shape[0]

    if args.mode == "stark":
        result = perturb.stark_shift(
            psi,
            potential,
            N=grid_size,
            L=args.L,
            Ez=args.Ez,
            steps=args.steps,
            dt=args.dt,
        )
        print(f"[STARK FIELD] Ez={args.Ez}")
        print(f"  → Numeric shift: {result.shifts['stark_shift_numeric']:.6e} hartree")
    elif args.mode == "zeeman":
        result = perturb.zeeman_shift(
            psi,
            potential,
            N=grid_size,
            L=args.L,
            Bz=args.Bz,
            steps=args.steps,
            dt=args.dt,
        )
        print(f"[ZEEMAN FIELD] Bz={args.Bz}")
        print(f"  → Numeric shift: {result.shifts['zeeman_shift_numeric']:.6e} hartree")
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    perturb.save_field_artifacts(result, out_dir, density_ref)
    print(f"  → Artifacts saved to {out_dir}")


def hyd_fstructure(args: argparse.Namespace) -> None:
    """Compute fine-structure corrections (Darwin, spin–orbit)."""

    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"Input directory {in_dir} not found")
        sys.exit(1)

    psi = np.load(in_dir / "psi.npy")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)

    n, l, m = args.nlm
    j = l + 0.5 if args.j is None else args.j

    grid_size = psi.shape[0]
    L = args.L
    dx = L / grid_size
    shifts = perturb.fine_structure(psi, Z=args.Z, n=n, l=l, j=j, dx=dx)

    print(f"[FINE STRUCTURE] (n={n}, l={l}, j={j})")
    print(f"  → Darwin term: {shifts['darwin_term']:.6e} hartree")
    print(f"  → Spin-orbit: {shifts['spin_orbit']:.6e} hartree")

    (out_dir / "shifts.json").write_text(json.dumps(shifts, indent=2))


def hyd_tomo(args: argparse.Namespace) -> None:
    """Run synthetic tomography on hydrogen density."""

    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"Input directory {in_dir} not found")
        sys.exit(1)

    density = np.load(in_dir / "density.npy")
    out_dir = Path(args.out)

    angles = np.linspace(0, np.pi, args.angles, endpoint=False)
    metrics = tomo.run_tomography(density, angles, noise=args.noise, out_dir=out_dir)

    print(f"[TOMOGRAPHY] angles={args.angles}, noise={args.noise}")
    print(f"  → SSIM: {metrics['ssim']:.4f}")
    print(f"  → PSNR: {metrics['psnr']:.2f} dB")
    print(f"  → L2 norm: {metrics['l2']:.4e}")
    print(f"  → Artifacts saved to {out_dir}")


def _parse_nlm(text: str):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--nlm requires three integers (n,l,m)")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main(argv=None):
    """Main entrypoint."""

    parser = argparse.ArgumentParser(
        description="3D Atom Modeling Suite (hydrogen, helium, fields, tomography)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hydrogen_cli.register_hydrogen_commands(subparsers)
    helium_cli.register_helium_commands(subparsers)

    p_field = subparsers.add_parser("hyd-field", help="Apply external field perturbation")
    p_field.add_argument("--mode", required=True, choices=["stark", "zeeman"])
    p_field.add_argument("--in", dest="input", required=True, help="Path to input directory")
    p_field.add_argument("--L", type=float, default=12.0, help="Box length (needed for k-space)")
    p_field.add_argument("--Ez", type=float, default=0.0, help="Electric field (Stark)")
    p_field.add_argument("--Bz", type=float, default=0.0, help="Magnetic field (Zeeman)")
    p_field.add_argument("--steps", type=int, default=500)
    p_field.add_argument("--dt", type=float, default=0.002)
    p_field.add_argument("--out", required=True)
    p_field.set_defaults(func=hyd_field)

    p_fs = subparsers.add_parser("hyd-fstructure", help="Fine structure corrections")
    p_fs.add_argument("--in", dest="input", required=True)
    p_fs.add_argument("--nlm", type=_parse_nlm, required=True, help="Quantum numbers (n,l,m)")
    p_fs.add_argument("--Z", type=float, default=1.0)
    p_fs.add_argument("--L", type=float, default=12.0, help="Box length (for dx calculation)")
    p_fs.add_argument("--j", type=float, default=None, help="Total angular momentum j")
    p_fs.add_argument("--out", required=True)
    p_fs.set_defaults(func=hyd_fstructure)

    p_tomo = subparsers.add_parser("hyd-tomo", help="Synthetic tomography")
    p_tomo.add_argument("--in", dest="input", required=True, help="Path to input directory with density.npy")
    p_tomo.add_argument("--angles", type=int, default=90, help="Number of projection angles")
    p_tomo.add_argument("--noise", type=float, default=0.0, help="Gaussian noise level")
    p_tomo.add_argument("--out", required=True)
    p_tomo.set_defaults(func=hyd_tomo)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
