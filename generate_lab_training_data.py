#!/usr/bin/env python3
"""
Generate JSONL training data from live Jarvis Lab API experiments.

This script calls the Jarvis Lab API to run thousands of quantum phase experiments
and turns them into instruction-style training samples for LLM fine-tuning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import requests

JARVIS_LAB_URL = "http://127.0.0.1:8000"

OUT_PATH = Path("data/lab_instructions.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def call(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Jarvis Lab API endpoint."""
    resp = requests.post(f"{JARVIS_LAB_URL}/{endpoint}", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def make_sample(instruction: str, response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Basic instruction-tuning sample schema."""
    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "meta": meta,
    }


def generate_phase_samples(n_per_phase: int = 30) -> List[Dict[str, Any]]:
    """Generate samples from single phase experiments."""
    phases = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]
    system_sizes = [16, 24, 32]
    depths = [4, 8, 12]
    biases = [None, 0.6, 0.7, 0.8]

    samples: List[Dict[str, Any]] = []

    for phase in phases:
        for _ in range(n_per_phase):
            params = {
                "phase_type": phase,
                "system_size": random.choice(system_sizes),
                "depth": random.choice(depths),
                "seed": random.randint(1, 100000),
                "bias": random.choice(biases),
            }
            result = call("run_phase_experiment", params)

            instruction = (
                f"Explain the behavior of a {phase} experiment with "
                f"system_size={params['system_size']}, depth={params['depth']}, "
                f"bias={params['bias']} (if given). Describe what the feature vector "
                f"and summary tell us about the phase."
            )

            summary = result.get("summary", {})
            fv = result.get("feature_vector", [])

            response = (
                "Here is an analysis of the experiment:\n\n"
                f"- Phase type: {phase}\n"
                f"- Parameters: {params}\n"
                f"- Feature vector (truncated): {fv[:8]} ...\n"
                f"- Summary stats: {summary}\n\n"
                "Interpretation:\n"
                "- Connect entropy / magnetization / correlations to what this phase means.\n"
                "- Describe whether this looks ordered, disordered, topological, or pseudorandom.\n"
                "- Mention how depth and bias influence the structure."
            )

            samples.append(make_sample(instruction, response, meta={"kind": "phase", "params": params}))
    return samples


def generate_tri_samples(n: int = 50) -> List[Dict[str, Any]]:
    """Generate samples from TRI (Time-Reversal Instability) experiments."""
    phases = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]
    samples: List[Dict[str, Any]] = []

    for _ in range(n):
        phase = random.choice(phases)
        params = {
            "phase_type": phase,
            "system_size": 32,
            "depth": random.choice([6, 8, 10, 12]),
            "bias": random.choice([0.6, 0.7, 0.8]),
            "seed": random.randint(1, 100000),
        }
        result = call("tri", params)

        tri = result.get("TRI")
        if tri is None:
            tri_display = "nan"
        else:
            tri_display = f"{tri:.4f}"
        instruction = (
            f"For phase '{phase}', we ran a Time-Reversal Instability (TRI) test with "
            f"bias={params['bias']} and depth={params['depth']}. "
            "Explain what the TRI value means and what it tells us about time-reversal "
            "fragility or symmetry of this phase."
        )

        response = (
            f"The TRI result for this configuration is TRI ≈ {tri_display}.\n\n"
            "Interpretation:\n"
            "- Explain what a high TRI vs low TRI implies.\n"
            "- Relate it to directional behavior / broken symmetries.\n"
            "- Compare how Ising vs SPT vs trivial vs pseudorandom usually behave."
        )

        samples.append(make_sample(instruction, response, meta={"kind": "tri", "params": params, "tri": tri}))
    return samples


def generate_discovery_samples(n_runs: int = 10) -> List[Dict[str, Any]]:
    """Generate samples from unsupervised discovery/clustering experiments."""
    samples: List[Dict[str, Any]] = []

    for _ in range(n_runs):
        phases = [
            "ising_symmetry_breaking",
            "spt_cluster",
            "trivial_product",
            "pseudorandom",
        ]
        payload = {
            "phases": phases,
            "num_per_phase": random.randint(10, 25),
            "k": len(phases),
            "iterations": 25,
        }
        result = call("discovery", payload)
        clusters = result.get("cluster_label_stats", [])

        instruction = (
            "We ran an unsupervised k-means clustering experiment on synthetic quantum phases.\n"
            f"Phases included: {phases}.\n"
            f"Cluster label stats: {clusters}.\n\n"
            "Explain what this tells us about how these phases group in feature space, "
            "which clusters are clean, where mixing happens, and what kind of hidden "
            "structure or phase boundaries this might reveal."
        )

        response = (
            "Interpretation of clustering results:\n"
            "- Identify which clusters are dominated by a single phase label.\n"
            "- Point out any clusters with significant mixing.\n"
            "- Hypothesize why some phases overlap in feature space.\n"
            "- Discuss how this validates or challenges the feature design."
        )

        samples.append(make_sample(instruction, response, meta={"kind": "discovery", "clusters": clusters}))
    return samples


def generate_replay_drift_samples(n: int = 30) -> List[Dict[str, Any]]:
    """Generate samples from replay drift scaling experiments."""
    phases = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]
    samples: List[Dict[str, Any]] = []

    for _ in range(n):
        phase = random.choice(phases)
        payload = {
            "phase_type": phase,
            "system_size": 32,
            "base_depth": random.choice([4, 6, 8]),
            "seed": random.randint(1, 100000),
            "depth_factors": [1, 2, 3, 4],
        }
        result = call("replay_drift", payload)
        runs = result.get("runs", [])

        instruction = (
            f"We ran replay drift scaling for phase '{phase}' with base depth "
            f"{payload['base_depth']} and depth factors {payload['depth_factors']}.\n"
            f"Runs (depth, drift): {[ (r['depth'], r['drift']) for r in runs ]}\n\n"
            "Explain how drift grows with depth for this phase and what that implies about "
            "its complexity, stability, or chaotic behavior."
        )

        response = (
            "Interpretation of replay drift:\n"
            "- Describe whether drift growth is roughly linear, sublinear, or superlinear.\n"
            "- Connect this to stability vs scrambling.\n"
            "- Compare how you expect this behavior to differ between trivial, SPT, Ising, and pseudorandom phases."
        )

        samples.append(make_sample(instruction, response, meta={"kind": "replay_drift", "runs": runs}))
    return samples


def main() -> None:
    print("🔬 Generating training data from live Jarvis Lab API experiments...")
    print(f"📡 Connecting to {JARVIS_LAB_URL}")
    print()

    all_samples: List[Dict[str, Any]] = []
    
    print("Phase experiments...")
    all_samples += generate_phase_samples(n_per_phase=40)
    print(f"  ✓ Generated {len(all_samples)} phase samples")
    
    print("TRI experiments...")
    tri_samples = generate_tri_samples(n=60)
    all_samples += tri_samples
    print(f"  ✓ Generated {len(tri_samples)} TRI samples")
    
    print("Discovery/clustering experiments...")
    discovery_samples = generate_discovery_samples(n_runs=15)
    all_samples += discovery_samples
    print(f"  ✓ Generated {len(discovery_samples)} discovery samples")
    
    print("Replay drift experiments...")
    drift_samples = generate_replay_drift_samples(n=40)
    all_samples += drift_samples
    print(f"  ✓ Generated {len(drift_samples)} replay drift samples")

    print()
    print(f"📊 Total: {len(all_samples)} training samples")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in all_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"💾 Saved to {OUT_PATH}")
    print()
    print("Next step: Run fine-tuning with finetune_ben_lab.py")


if __name__ == "__main__":
    main()
