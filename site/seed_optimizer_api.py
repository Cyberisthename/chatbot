#!/usr/bin/env python3
"""
Seed Optimizer API — web-demo bridge for the FBSC Variational Seed Optimizer.
================================================================================
Called by src/routes/api/optimize.ts (TanStack route) with argv, prints one JSON
doc to stdout. Mirrors compressor_api.py conventions.

Usage:
  python3 seed_optimizer_api.py <objective> <seed1> <seed2> <seed3> [qubits] [generations] [population]
"""
import sys
import json
from pathlib import Path

sys.path.append("/home/team/shared/chatbot")

try:
    from variational_seed_optimizer import optimize_seed, DEFAULT_BOUNDS
    HAS_VSO = True
except Exception as e:  # pragma: no cover
    HAS_VSO = False
    IMPORT_ERR = str(e)


def run_optimizer(objective, seed_vals, qubits=128, generations=40, population=18):
    if not HAS_VSO:
        return {"error": f"optimizer unavailable: {IMPORT_ERR}"}
    try:
        report = optimize_seed(
            seed=tuple(float(v) for v in seed_vals),
            objective=objective,
            n_effective_qubits=int(qubits),
            generations=int(generations),
            population=int(population),
            bounds=DEFAULT_BOUNDS,
            polish=True,
            outdir="/home/team/shared/chatbot/artifacts/seed_optimizer/web",
            verbose=False,
        )
        # Slim the payload for the web (drop full history; keep last 60 points).
        slim = {
            "objective": report["objective"],
            "owner_seed_before": report["owner_seed_before"],
            "optimized_seed": report["optimized_seed"],
            "before": report["before"],
            "after": report["after"],
            "improvement_pct": report["improvement_pct"],
            "bio_resonance_before": report["bio_resonance_before"],
            "bio_resonance_after": report["bio_resonance_after"],
            "params": report["params"],
            "reconstruction": report["reconstruction"],
            "convergence": report["convergence"][-60:],
            "verified_fresh_instance": report["verified_fresh_instance"],
        }
        return slim
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(json.dumps({"error": "usage: seed_optimizer_api.py <objective> <s1> <s2> <s3> [qubits] [generations] [population]"}))
        sys.exit(1)
    try:
        objective = sys.argv[1]
        seeds = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
        qubits = int(sys.argv[5]) if len(sys.argv) > 5 else 128
        gens = int(sys.argv[6]) if len(sys.argv) > 6 else 40
        pop = int(sys.argv[7]) if len(sys.argv) > 7 else 18
        print(json.dumps(run_optimizer(objective, seeds, qubits, gens, pop)))
    except Exception as e:  # pragma: no cover
        print(json.dumps({"error": str(e)}))