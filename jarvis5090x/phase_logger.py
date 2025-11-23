from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .quantum_layer import Branch

SNAPSHOT_LIMIT = 32


@dataclass
class LayerLog:
    layer_idx: int
    event_type: str
    branch_count: int
    probabilities: List[float]
    probability_entropy: float
    snapshot: List[Dict[str, Any]]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    experiment_id: str
    phase_type: str
    params: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    layers: List[LayerLog] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    feature_vector: Optional[List[float]] = None


class PhaseLogger:
    def __init__(self) -> None:
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._snapshot_limit = SNAPSHOT_LIMIT

    def start_experiment(
        self,
        experiment_id: str,
        phase_type: str,
        params: Dict[str, Any],
    ) -> None:
        self._experiments[experiment_id] = ExperimentRecord(
            experiment_id=experiment_id,
            phase_type=phase_type,
            params=copy.deepcopy(params),
            start_time=time.time(),
        )

    def log_spawn(
        self,
        experiment_id: str,
        branches: List[Branch],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._log_layer(experiment_id, "spawn", branches, metadata or {})

    def log_interfere(
        self,
        experiment_id: str,
        branches: List[Branch],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._log_layer(experiment_id, "interfere", branches, metadata or {})

    def log_collapse(
        self,
        experiment_id: str,
        branches: List[Branch],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._log_layer(experiment_id, "collapse", branches, metadata or {})

    def log_final_state(
        self,
        experiment_id: str,
        collapsed_state: Dict[str, Any],
        summary_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._experiments.get(experiment_id)
        if record is None:
            return

        record.final_state = copy.deepcopy(collapsed_state)
        record.end_time = time.time()
        record.summary_stats = summary_stats or self._summarize_state(collapsed_state)

    def attach_feature_vector(self, experiment_id: str, feature_vector: List[float]) -> None:
        record = self._experiments.get(experiment_id)
        if record is None:
            return
        record.feature_vector = list(feature_vector)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self._experiments.get(experiment_id)

    def iter_experiments(self) -> Iterable[ExperimentRecord]:
        return tuple(self._experiments.values())

    def list_experiments(self) -> List[str]:
        return list(self._experiments.keys())

    def clear(self) -> None:
        self._experiments.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "total_experiments": len(self._experiments),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_layer(
        self,
        experiment_id: str,
        event_type: str,
        branches: List[Branch],
        metadata: Dict[str, Any],
    ) -> None:
        record = self._experiments.get(experiment_id)
        if record is None:
            return

        layer_idx = int(metadata.get("layer_index", len(record.layers)))
        raw_probabilities = [branch.probability() for branch in branches]
        probabilities = self._normalize_probabilities(raw_probabilities)
        entropy = self._probability_entropy(probabilities)
        snapshot = self._capture_snapshot(branches, probabilities)

        layer_log = LayerLog(
            layer_idx=layer_idx,
            event_type=event_type,
            branch_count=len(branches),
            probabilities=probabilities,
            probability_entropy=entropy,
            snapshot=snapshot,
            timestamp=time.time(),
            metadata=copy.deepcopy(metadata),
        )
        record.layers.append(layer_log)

    def _normalize_probabilities(self, probabilities: List[float]) -> List[float]:
        total = sum(probabilities)
        if total <= 0:
            return [0.0 for _ in probabilities]
        return [p / total for p in probabilities]

    def _probability_entropy(self, probabilities: List[float]) -> float:
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p + 1e-12, 2)
        return entropy

    def _capture_snapshot(
        self,
        branches: List[Branch],
        probabilities: List[float],
    ) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        limit = min(self._snapshot_limit, len(branches))
        for idx in range(limit):
            branch = branches[idx]
            probability = probabilities[idx] if idx < len(probabilities) else 0.0
            snapshot.append(
                {
                    "probability": probability,
                    "amplitude": {
                        "real": float(branch.amplitude.real),
                        "imag": float(branch.amplitude.imag),
                    },
                    "state": copy.deepcopy(branch.state),
                }
            )
        return snapshot

    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        spins = state.get("spins")
        if isinstance(spins, list) and spins:
            summary["magnetization"] = float(self._magnetization(spins))
            summary["entropy_proxy"] = float(self._entropy_proxy(spins))
            summary["correlation_length"] = float(self._correlation_length(spins))

        for key in (
            "magnetization",
            "entropy_proxy",
            "correlation_length",
            "symmetry_indicator",
            "string_order",
            "randomness_score",
        ):
            value = state.get(key)
            if isinstance(value, (int, float)):
                summary[key] = float(value)

        return summary

    def _magnetization(self, spins: List[Any]) -> float:
        if not spins:
            return 0.0
        total = 0.0
        for spin in spins:
            try:
                total += float(spin)
            except (TypeError, ValueError):
                total += 0.0
        return total / len(spins)

    def _entropy_proxy(self, spins: List[Any]) -> float:
        if len(spins) < 2:
            return 0.0
        transitions = 0
        for idx, spin in enumerate(spins):
            try:
                current = float(spin)
                nxt = float(spins[(idx + 1) % len(spins)])
            except (TypeError, ValueError):
                continue
            if current != nxt:
                transitions += 1
        return transitions / len(spins)

    def _correlation_length(self, spins: List[Any]) -> float:
        if len(spins) < 2:
            return 0.0
        max_distance = min(len(spins) // 2, 8)
        if max_distance <= 0:
            return 0.0
        total = 0.0
        weight = 0.0
        for distance in range(1, max_distance + 1):
            correlation = 0.0
            count = 0
            for idx in range(len(spins)):
                try:
                    correlation += float(spins[idx]) * float(spins[(idx + distance) % len(spins)])
                    count += 1
                except (TypeError, ValueError):
                    continue
            if count == 0:
                continue
            correlation /= count
            total += abs(correlation) * distance
            weight += distance
        if weight <= 0:
            return 0.0
        return total / weight
