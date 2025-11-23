from __future__ import annotations

import math
from typing import Any, Dict, List

from .phase_logger import ExperimentRecord, LayerLog


def compute_probability_entropy_profile(record: ExperimentRecord) -> List[float]:
    profile = []
    for layer_log in record.layers:
        if not layer_log.probabilities:
            profile.append(0.0)
            continue

        total = sum(layer_log.probabilities)
        if total <= 0:
            profile.append(0.0)
            continue

        normalized_probs = [p / total for p in layer_log.probabilities]
        entropy = -sum(
            p * math.log2(p) if p > 0 else 0.0 for p in normalized_probs
        )
        profile.append(entropy)

    return profile


def compute_branch_count_profile(record: ExperimentRecord) -> List[int]:
    return [layer_log.branch_count for layer_log in record.layers]


def compute_scrambling_score(record: ExperimentRecord) -> float:
    if not record.layers:
        return 0.0

    uniformity_scores = []
    for layer_log in record.layers:
        if not layer_log.probabilities or layer_log.branch_count == 0:
            continue

        total = sum(layer_log.probabilities)
        if total <= 0:
            continue

        normalized = [p / total for p in layer_log.probabilities]
        uniform_prob = 1.0 / layer_log.branch_count
        variance = sum((p - uniform_prob) ** 2 for p in normalized) / layer_log.branch_count
        uniformity = 1.0 / (1.0 + variance * 10)
        uniformity_scores.append(uniformity)

    if not uniformity_scores:
        return 0.0

    return sum(uniformity_scores) / len(uniformity_scores)


def compute_correlation_profile(record: ExperimentRecord) -> List[float]:
    profile = []
    for layer_log in record.layers:
        if layer_log.branch_count <= 1:
            profile.append(0.0)
            continue

        correlation = math.log(layer_log.branch_count) / math.log(2)
        profile.append(correlation)

    return profile


def extract_features(record: ExperimentRecord) -> List[float]:
    entropy_profile = compute_probability_entropy_profile(record)
    branch_profile = compute_branch_count_profile(record)
    scrambling = compute_scrambling_score(record)
    correlation_profile = compute_correlation_profile(record)

    features: List[float] = []

    if entropy_profile:
        features.append(sum(entropy_profile) / len(entropy_profile))
        features.append(max(entropy_profile))
        features.append(min(entropy_profile))
        features.append(entropy_profile[-1] if entropy_profile else 0.0)
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    if branch_profile:
        features.append(sum(branch_profile) / len(branch_profile))
        features.append(float(max(branch_profile)))
        features.append(float(min(branch_profile)))
        features.append(float(branch_profile[-1]))
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    features.append(scrambling)

    if correlation_profile:
        features.append(sum(correlation_profile) / len(correlation_profile))
        features.append(max(correlation_profile))
        features.append(min(correlation_profile))
    else:
        features.extend([0.0, 0.0, 0.0])

    features.append(float(len(record.layers)))

    if record.end_time and record.start_time:
        features.append(record.end_time - record.start_time)
    else:
        features.append(0.0)

    system_size = record.params.get("system_size", 0)
    depth = record.params.get("depth", 0)
    features.append(float(system_size))
    features.append(float(depth))

    return features


def feature_names() -> List[str]:
    return [
        "entropy_mean",
        "entropy_max",
        "entropy_min",
        "entropy_final",
        "branch_count_mean",
        "branch_count_max",
        "branch_count_min",
        "branch_count_final",
        "scrambling_score",
        "correlation_mean",
        "correlation_max",
        "correlation_min",
        "layer_count",
        "execution_time",
        "system_size",
        "depth",
    ]
