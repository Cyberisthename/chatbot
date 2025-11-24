from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis5090x import (
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
    PhaseDetector,
)


FeatureVector = Sequence[float]


def make_detector() -> PhaseDetector:
    devices = [
        AdapterDevice(
            id="quantum_0",
            label="Quantum Simulator",
            kind=DeviceKind.VIRTUAL,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.QUANTUM},
        ),
    ]
    orchestrator = Jarvis5090X(devices)
    detector = PhaseDetector(orchestrator)
    return detector


def l2_distance(a: FeatureVector, b: FeatureVector) -> float:
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def run_time_reversal_test(
    detector: PhaseDetector,
    phase_type: str = "ising_symmetry_breaking",
    system_size: int = 32,
    depth: int = 8,
    bias: float = 0.7,
    seed: int = 42,
) -> Dict[str, Any]:
    forward = detector.run_phase_experiment(
        phase_type=phase_type,
        system_size=system_size,
        depth=depth,
        seed=seed,
        bias=bias,
    )
    reverse_bias = 1.0 - bias
    reverse = detector.run_phase_experiment(
        phase_type=phase_type,
        system_size=system_size,
        depth=depth,
        seed=seed,
        bias=reverse_bias,
    )

    forward_vec = forward["feature_vector"]
    reverse_vec = reverse["feature_vector"]
    tri = l2_distance(forward_vec, reverse_vec)

    return {
        "forward_id": forward["experiment_id"],
        "reverse_id": reverse["experiment_id"],
        "forward_features": forward_vec,
        "reverse_features": reverse_vec,
        "TRI": tri,
        "params": {
            "phase_type": phase_type,
            "system_size": system_size,
            "depth": depth,
            "bias": bias,
            "reverse_bias": reverse_bias,
            "seed": seed,
        },
    }


def build_mixed_phase_samples(
    detector: PhaseDetector,
    *,
    phases: Sequence[str],
    num_per_phase: int = 20,
    system_size: int = 32,
    depth: int = 8,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for phase in phases:
        for _ in range(num_per_phase):
            seed = random.randint(1, 10_000)
            result = detector.run_phase_experiment(
                phase_type=phase,
                system_size=system_size,
                depth=depth,
                seed=seed,
            )
            samples.append(
                {
                    "phase": phase,
                    "features": result["feature_vector"],
                    "id": result["experiment_id"],
                }
            )
    return samples


def _euclidean(a: FeatureVector, b: FeatureVector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_vector(vectors: Iterable[FeatureVector]) -> List[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    dims = len(vectors[0])
    return [
        sum(vec[idx] for vec in vectors) / len(vectors)
        for idx in range(dims)
    ]


def kmeans(
    vectors: Sequence[FeatureVector],
    *,
    k: int = 4,
    iterations: int = 20,
) -> Tuple[List[List[float]], List[List[int]]]:
    if not vectors:
        raise ValueError("Cannot cluster empty vector set")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(vectors):
        raise ValueError("k cannot exceed number of vectors")

    centroids = [list(vec) for vec in random.sample(list(vectors), k)]
    assignments: List[List[int]] = [[] for _ in range(k)]

    for _ in range(iterations):
        assignments = [[] for _ in range(k)]
        for idx, vec in enumerate(vectors):
            distances = [_euclidean(vec, centroid) for centroid in centroids]
            best = min(range(k), key=lambda i: distances[i])
            assignments[best].append(idx)

        new_centroids: List[List[float]] = []
        for cluster_indices in assignments:
            if not cluster_indices:
                new_centroids.append(list(random.choice(vectors)))
                continue
            cluster_vectors = [vectors[i] for i in cluster_indices]
            new_centroids.append(_mean_vector(cluster_vectors))
        centroids = new_centroids

    return centroids, assignments


def unsupervised_phase_discovery(
    detector: PhaseDetector,
    *,
    phases: Sequence[str],
    num_per_phase: int = 20,
    k: int = 4,
    iterations: int = 25,
) -> Dict[str, Any]:
    samples = build_mixed_phase_samples(
        detector,
        phases=phases,
        num_per_phase=num_per_phase,
    )
    vectors = [sample["features"] for sample in samples]
    centroids, assignments = kmeans(vectors, k=k, iterations=iterations)

    cluster_label_stats: List[Dict[str, int]] = []
    for cluster_indices in assignments:
        stats: Dict[str, int] = {}
        for idx in cluster_indices:
            phase = samples[idx]["phase"]
            stats[phase] = stats.get(phase, 0) + 1
        cluster_label_stats.append(dict(sorted(stats.items())))

    return {
        "centroids": centroids,
        "assignments": assignments,
        "samples": samples,
        "cluster_label_stats": cluster_label_stats,
    }


def replay_drift_scaling(
    detector: PhaseDetector,
    *,
    phase_type: str = "ising_symmetry_breaking",
    system_size: int = 32,
    base_depth: int = 6,
    seed: int = 123,
    depth_factors: Sequence[int] = (1, 2, 3),
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for factor in depth_factors:
        depth = base_depth * factor
        result = detector.run_phase_experiment(
            phase_type=phase_type,
            system_size=system_size,
            depth=depth,
            seed=seed,
        )
        runs.append(
            {
                "depth": depth,
                "id": result["experiment_id"],
                "features": result["feature_vector"],
            }
        )

    if not runs:
        return runs

    base_features = runs[0]["features"]
    for run in runs:
        run["drift"] = l2_distance(base_features, run["features"])
    return runs


def main() -> None:
    random.seed(1337)
    detector = make_detector()

    tracked_phases = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]

    print("\n=== EXPERIMENT A: Time-Reversal Instability ===")
    for phase in tracked_phases:
        result = run_time_reversal_test(detector, phase_type=phase)
        print(f"{phase:25s} TRI = {result['TRI']:.6f}")

    print("\n=== EXPERIMENT B: Unsupervised Phase Discovery ===")
    discovery = unsupervised_phase_discovery(
        detector,
        phases=tracked_phases,
        num_per_phase=30,
        k=len(tracked_phases),
    )
    for idx, stats in enumerate(discovery["cluster_label_stats"]):
        print(f"Cluster {idx}: {stats}")

    print("\n=== EXPERIMENT C: Replay Drift Scaling ===")
    for phase in tracked_phases:
        runs = replay_drift_scaling(
            detector,
            phase_type=phase,
            system_size=32,
            base_depth=4,
            seed=999,
            depth_factors=(1, 2, 3, 4),
        )
        print(f"\nPhase: {phase}")
        for run in runs:
            print(f"  depth={run['depth']:3d}  drift={run['drift']:.6f}")


if __name__ == "__main__":
    main()
