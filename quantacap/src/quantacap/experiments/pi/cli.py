"""CLI bridge for π-phase extension experiments."""

from __future__ import annotations

import argparse
import json

from .couple import run_pi_coupling
from .drift import run_pi_drift
from .noise import run_pi_noise_scan, run_pi_entropy_collapse
from .entropy import run_pi_entropy_control


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="π-phase extension suite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    couple = sub.add_parser("couple", help="Couple two π-locked oscillators")
    couple.add_argument("--kappa", type=float, required=True)
    couple.add_argument("--steps", type=int, default=50000)

    drift = sub.add_parser("drift", help="Material-time drift simulation")
    drift.add_argument("--rate", type=float, required=True)
    drift.add_argument("--steps", type=int, default=100000)

    noise = sub.add_parser("noise", help="Noise-collapse sweep")
    noise.add_argument("--sigma-max", type=float, required=True)
    noise.add_argument("--sigma-min", type=float, default=0.0)
    noise.add_argument("--steps", type=int, default=41)
    noise.add_argument("--rotations", type=int, default=1000)
    noise.add_argument("--entropy-threshold", type=float, default=0.05)
    noise.add_argument("--entropy-bins", type=int, default=64)
    noise.add_argument("--schedule", choices=["linear", "log"], default="linear")
    noise.add_argument("--seed", type=int, default=424242)
    noise.add_argument("--out", type=str, default=None)
    noise.add_argument("--adapter-id", type=str, default=None)
    noise.add_argument("--plateau-epsilon", type=float, default=1e-3)
    noise.add_argument("--no-detect-steps", action="store_true")
    noise.add_argument("--no-entropy", action="store_true")

    collapse = sub.add_parser(
        "collapse", help="Synthetic entropy collapse scan with coupled oscillators"
    )
    collapse.add_argument("--kappa", type=float, default=0.02)
    collapse.add_argument("--sigma-min", type=float, default=1e-9)
    collapse.add_argument("--sigma-max", type=float, default=1e-6)
    collapse.add_argument("--stages", type=int, default=25)
    collapse.add_argument("--stage-length", type=int, default=128)
    collapse.add_argument("--entropy-threshold", type=float, default=1e-3)

    entropy = sub.add_parser("entropy", help="Entropy minimisation control loop")
    entropy.add_argument("--steps", type=int, default=80000)

    args = parser.parse_args(argv)

    if args.cmd == "couple":
        result = run_pi_coupling(kappa=args.kappa, steps=args.steps)
    elif args.cmd == "drift":
        result = run_pi_drift(rate=args.rate, steps=args.steps)
    elif args.cmd == "noise":
        result = run_pi_noise_scan(
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            steps=args.steps,
            rotations=args.rotations,
            entropy_threshold=args.entropy_threshold,
            entropy_bins=args.entropy_bins,
            schedule=args.schedule,
            seed=args.seed,
            artifact_path=args.out,
            adapter_id=args.adapter_id,
            detect_steps=not args.no_detect_steps,
            track_entropy=not args.no_entropy,
            plateau_epsilon=args.plateau_epsilon,
        )
    elif args.cmd == "entropy":
        result = run_pi_entropy_control(steps=args.steps)
    elif args.cmd == "collapse":
        result = run_pi_entropy_collapse(
            kappa=args.kappa,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            stages=args.stages,
            stage_length=args.stage_length,
            entropy_threshold=args.entropy_threshold,
        )
    else:  # pragma: no cover - defensive
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
