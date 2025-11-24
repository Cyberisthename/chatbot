from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .orchestrator import Jarvis5090X
from .phase_classifier import CentroidPhaseClassifier
from .phase_dataset import PhaseDataset, dataset_from_records
from .phase_features import extract_features
from .phase_mlp_classifier import MLPPhaseClassifier
from .phase_logger import ExperimentRecord, PhaseLogger
from .phase_replay import PhaseReplay
from .quantum_layer import QuantumExperimentHooks


@dataclass
class PhaseExperimentConfig:
    base_state: Dict[str, Any]
    variations: List[Dict[str, Any]]
    scoring_key: Optional[str] = "score"
    scoring_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentHandle:
    experiment_id: str
    phase_type: str
    params: Dict[str, Any]
    signature: str
    feature_vector: List[float]
    summary: Dict[str, Any]
    collapsed_state: Dict[str, Any]


PhaseGenerator = Callable[[int, int, int, Dict[str, Any]], PhaseExperimentConfig]


# ---------------------------------------------------------------------------
# Helper functions for synthetic phase generation
# ---------------------------------------------------------------------------


def _magnetization(spins: List[Any]) -> float:
    if not spins:
        return 0.0
    total = 0.0
    for value in spins:
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total / len(spins)


def _entropy_proxy(spins: List[Any]) -> float:
    if len(spins) < 2:
        return 0.0
    transitions = 0
    for idx, value in enumerate(spins):
        try:
            current = float(value)
            nxt = float(spins[(idx + 1) % len(spins)])
        except (TypeError, ValueError):
            continue
        if current != nxt:
            transitions += 1
    return transitions / len(spins)


def _correlation_length(spins: List[Any], max_distance: int = 8) -> float:
    if len(spins) < 2:
        return 0.0
    capped_distance = min(max_distance, len(spins) // 2)
    if capped_distance <= 0:
        return 0.0
    total = 0.0
    weight = 0.0
    for distance in range(1, capped_distance + 1):
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


def _string_order(spins: List[Any]) -> float:
    if len(spins) < 3:
        return 0.0
    order_total = 0.0
    count = 0
    for idx in range(len(spins) - 2):
        try:
            order_total += float(spins[idx]) * float(spins[idx + 1]) * float(spins[idx + 2])
            count += 1
        except (TypeError, ValueError):
            continue
    if count == 0:
        return 0.0
    return order_total / count


def _randomness_score(spins: List[Any]) -> float:
    entropy = _entropy_proxy(spins)
    return min(1.0, 0.4 + entropy)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Phase generator implementations
# ---------------------------------------------------------------------------


def _ising_generator(system_size: int, depth: int, seed: int, options: Dict[str, Any]) -> PhaseExperimentConfig:
    rng = random.Random(seed)
    bias = float(options.get("bias", 0.65))
    base_spins = [1 if rng.random() < bias else -1 for _ in range(system_size)]
    base_magnetization = _magnetization(base_spins)
    base_entropy = _entropy_proxy(base_spins)
    base_state = {
        "phase": "ising_symmetry_breaking",
        "spins": base_spins,
        "magnetization": base_magnetization,
        "entropy_proxy": base_entropy,
        "correlation_length": _correlation_length(base_spins),
        "string_order": _string_order(base_spins) * 0.1,
        "symmetry_indicator": 1.0 if abs(base_magnetization) > 0.35 else 0.0,
        "randomness_score": _randomness_score(base_spins) * 0.5,
        "score": 1.0 + abs(base_magnetization) + 0.1 * system_size,
    }

    variations: List[Dict[str, Any]] = []
    for layer in range(depth):
        spins = list(base_spins)
        domain_size = max(1, int(system_size * (0.08 + 0.12 * rng.random())))
        start_idx = rng.randrange(system_size)
        for offset in range(domain_size):
            idx = (start_idx + offset) % system_size
            if rng.random() < 0.45:
                spins[idx] *= -1
        magnetization = _magnetization(spins)
        correlation = _correlation_length(spins)
        entropy = _entropy_proxy(spins)
        variations.append(
            {
                "layer": layer,
                "phase": "ising_symmetry_breaking",
                "spins": spins,
                "magnetization": magnetization,
                "correlation_length": correlation,
                "entropy_proxy": entropy,
                "symmetry_indicator": 1.0 if abs(magnetization) > 0.3 else 0.0,
                "string_order": _string_order(spins) * 0.1,
                "randomness_score": _randomness_score(spins),
                "score": 1.0 + abs(magnetization) + 0.2 * correlation,
            }
        )

    metadata = {"phase_family": "ising", "symmetry_breaking": True}
    return PhaseExperimentConfig(base_state=base_state, variations=variations, metadata=metadata)


def _spt_cluster_generator(system_size: int, depth: int, seed: int, options: Dict[str, Any]) -> PhaseExperimentConfig:
    rng = random.Random(seed + 137)
    base_spins = [1 if (idx + seed) % 2 == 0 else -1 for idx in range(system_size)]
    base_string_order = 0.85
    base_state = {
        "phase": "spt_cluster",
        "spins": base_spins,
        "magnetization": _magnetization(base_spins),
        "entropy_proxy": _entropy_proxy(base_spins) * 0.4,
        "correlation_length": _correlation_length(base_spins),
        "string_order": base_string_order,
        "symmetry_indicator": 1.0,
        "randomness_score": 0.35,
        "edge_mode_imbalance": 0.8,
        "is_spt": 1.0,
        "score": 1.0 + base_string_order,
    }

    variations: List[Dict[str, Any]] = []
    for layer in range(depth):
        spins = list(base_spins)
        toggle_rate = 0.12 + 0.04 * rng.random()
        for idx in range(system_size):
            if rng.random() < toggle_rate:
                spins[idx] *= -1
        string_order = max(0.4, base_string_order - 0.04 * layer + 0.05 * rng.random())
        entropy = _entropy_proxy(spins) * 0.6
        variations.append(
            {
                "layer": layer,
                "phase": "spt_cluster",
                "spins": spins,
                "magnetization": _magnetization(spins),
                "correlation_length": _correlation_length(spins),
                "entropy_proxy": entropy,
                "string_order": string_order,
                "symmetry_indicator": 1.0,
                "edge_mode_imbalance": 0.75 + 0.05 * rng.random(),
                "randomness_score": 0.35 + entropy * 0.4,
                "is_spt": 1.0,
                "score": 1.0 + string_order + 0.1 * system_size,
            }
        )

    metadata = {"phase_family": "spt", "topological": True}
    return PhaseExperimentConfig(base_state=base_state, variations=variations, metadata=metadata)


def _trivial_product_generator(system_size: int, depth: int, seed: int, options: Dict[str, Any]) -> PhaseExperimentConfig:
    rng = random.Random(seed + 271)
    base_spins = [1 if rng.random() < 0.55 else -1 for _ in range(system_size)]
    base_entropy = _entropy_proxy(base_spins) * 0.4
    base_state = {
        "phase": "trivial_product",
        "spins": base_spins,
        "magnetization": _magnetization(base_spins),
        "entropy_proxy": base_entropy,
        "correlation_length": _correlation_length(base_spins) * 0.6,
        "string_order": 0.05,
        "symmetry_indicator": 0.2,
        "randomness_score": 0.25,
        "score": 1.0 + 0.3 * (1.0 - base_entropy),
    }

    variations: List[Dict[str, Any]] = []
    for layer in range(depth):
        spins = list(base_spins)
        for idx in range(system_size):
            if rng.random() < 0.05:
                spins[idx] *= -1
        entropy = _entropy_proxy(spins) * 0.5
        variations.append(
            {
                "layer": layer,
                "phase": "trivial_product",
                "spins": spins,
                "magnetization": _magnetization(spins),
                "correlation_length": _correlation_length(spins) * 0.6,
                "entropy_proxy": entropy,
                "string_order": 0.05,
                "symmetry_indicator": 0.2,
                "randomness_score": 0.2 + entropy * 0.5,
                "score": 1.0 + 0.5 * (1.0 - entropy),
            }
        )

    metadata = {"phase_family": "product", "correlated": False}
    return PhaseExperimentConfig(base_state=base_state, variations=variations, metadata=metadata)


def _pseudorandom_generator(system_size: int, depth: int, seed: int, options: Dict[str, Any]) -> PhaseExperimentConfig:
    rng = random.Random(seed + 911)
    base_spins = [1 if rng.random() < 0.5 else -1 for _ in range(system_size)]
    base_entropy = _entropy_proxy(base_spins)
    base_state = {
        "phase": "pseudorandom",
        "spins": base_spins,
        "magnetization": _magnetization(base_spins),
        "entropy_proxy": base_entropy,
        "correlation_length": _correlation_length(base_spins),
        "string_order": _string_order(base_spins) * 0.05,
        "symmetry_indicator": 0.1,
        "randomness_score": 0.6 + 0.4 * base_entropy,
        "score": 1.0 + 0.8 * base_entropy,
    }

    variations: List[Dict[str, Any]] = []
    for layer in range(depth):
        spins = [1 if rng.random() < 0.5 else -1 for _ in range(system_size)]
        entropy = _entropy_proxy(spins)
        variations.append(
            {
                "layer": layer,
                "phase": "pseudorandom",
                "spins": spins,
                "magnetization": _magnetization(spins),
                "correlation_length": _correlation_length(spins),
                "entropy_proxy": entropy,
                "string_order": _string_order(spins) * 0.05,
                "symmetry_indicator": 0.1,
                "randomness_score": 0.7 + 0.3 * entropy,
                "score": 1.0 + entropy,
            }
        )

    metadata = {"phase_family": "pseudorandom", "scrambling": True}
    return PhaseExperimentConfig(base_state=base_state, variations=variations, metadata=metadata)


DEFAULT_GENERATORS: Dict[str, PhaseGenerator] = {
    "ising_symmetry_breaking": _ising_generator,
    "spt_cluster": _spt_cluster_generator,
    "trivial_product": _trivial_product_generator,
    "pseudorandom": _pseudorandom_generator,
}


# ---------------------------------------------------------------------------
# PhaseDetector implementation
# ---------------------------------------------------------------------------


class PhaseDetector:
    def __init__(
        self,
        orchestrator: Jarvis5090X,
        logger: Optional[PhaseLogger] = None,
        generators: Optional[Dict[str, PhaseGenerator]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.logger = logger or PhaseLogger()
        self.generators = dict(generators or DEFAULT_GENERATORS)
        self.registry: Dict[str, ExperimentHandle] = {}
        self.replay_engine = PhaseReplay(self.logger)
        self.centroid_classifier = CentroidPhaseClassifier()
        self.mlp_classifier: Optional[MLPPhaseClassifier]
        try:
            self.mlp_classifier = MLPPhaseClassifier()
        except RuntimeError:
            self.mlp_classifier = None
        self.classifier = self.centroid_classifier
        self._classifier_trained = False
        self._centroid_classifier_trained = False
        self._mlp_classifier_trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def use_mlp_classifier(self) -> None:
        if self.mlp_classifier is None or not self.mlp_classifier.is_available():
            raise RuntimeError(
                "MLPPhaseClassifier unavailable. Install torch to enable the neural classifier."
            )
        self.classifier = self.mlp_classifier
        self._classifier_trained = self._mlp_classifier_trained

    def use_centroid_classifier(self) -> None:
        self.classifier = self.centroid_classifier
        self._classifier_trained = self._centroid_classifier_trained

    def run_phase_experiment(
        self,
        phase_type: str,
        system_size: int,
        depth: int,
        seed: int,
        *,
        top_k: int = 1,
        **phase_options: Any,
    ) -> Dict[str, Any]:
        if phase_type not in self.generators:
            raise ValueError(f"Unknown phase type: {phase_type}")

        result = self._execute_phase_experiment(
            phase_type=phase_type,
            system_size=system_size,
            depth=depth,
            seed=seed,
            top_k=top_k,
            phase_options=phase_options,
        )

        experiment_id = result["experiment_id"]
        feature_vector = result["feature_vector"]
        summary = result["summary"]

        handle = ExperimentHandle(
            experiment_id=experiment_id,
            phase_type=phase_type,
            params=result["params"],
            signature=result["signature"],
            feature_vector=feature_vector,
            summary=summary,
            collapsed_state=result["collapsed_state"],
        )
        self.registry[experiment_id] = handle
        return {
            "experiment_id": experiment_id,
            "phase_type": phase_type,
            "feature_vector": feature_vector,
            "summary": summary,
            "result": result["raw_result"],
        }

    def log_phase_features(self, experiment_id: str) -> List[float]:
        record = self.logger.get_experiment(experiment_id)
        if record is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if record.feature_vector is None:
            feature_vector = extract_features(record)
            self.logger.attach_feature_vector(experiment_id, feature_vector)
        return list(record.feature_vector or [])

    def replay_experiment(
        self,
        experiment_id: str,
        *,
        compare: bool = True,
    ) -> Dict[str, Any]:
        original = self.logger.get_experiment(experiment_id)
        if original is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        params = dict(original.params)
        phase_type = original.phase_type
        system_size = int(params.get("system_size", 0))
        depth = int(params.get("depth", 0))
        seed = int(params.get("seed", 0))
        top_k = int(params.get("top_k", 1))
        extras = {k: v for k, v in params.items() if k not in {"system_size", "depth", "seed", "top_k"}}

        replay_id = f"{experiment_id}::replay::{uuid.uuid4().hex[:6]}"
        result = self._execute_phase_experiment(
            phase_type=phase_type,
            system_size=system_size,
            depth=depth,
            seed=seed,
            top_k=top_k,
            phase_options=extras,
            experiment_id=replay_id,
            register=False,
        )

        comparison = None
        if compare:
            comparison = self.replay_engine.compare_experiments(experiment_id, replay_id)

        return {
            "experiment_id": experiment_id,
            "replay_id": replay_id,
            "feature_vector": result["feature_vector"],
            "summary": result["summary"],
            "comparison": comparison,
            "result": result["raw_result"],
        }

    def classify_phase(
        self,
        *,
        experiment_id: Optional[str] = None,
        feature_vector: Optional[List[float]] = None,
        retrain: bool = False,
        dataset: Optional[PhaseDataset] = None,
    ) -> Dict[str, Any]:
        if feature_vector is None:
            if not experiment_id:
                raise ValueError("Provide experiment_id or feature_vector")
            feature_vector = self.log_phase_features(experiment_id)

        dataset = dataset or dataset_from_records(self.logger.iter_experiments())
        if not dataset.examples:
            raise ValueError("No dataset available for classification")

        if self.classifier is self.mlp_classifier:
            if self.mlp_classifier is None or not self.mlp_classifier.is_available():
                raise RuntimeError(
                    "MLPPhaseClassifier unavailable. Install torch or switch to centroid classifier."
                )

        training_report = None
        if retrain or not self._classifier_trained:
            training_report = self.classifier.train(dataset)
            self._classifier_trained = True
            if self.classifier is self.centroid_classifier:
                self._centroid_classifier_trained = True
            elif self.classifier is self.mlp_classifier:
                self._mlp_classifier_trained = True

        label, confidence = self.classifier.predict(feature_vector)
        return {
            "prediction": label,
            "confidence": confidence,
            "classifier": self.classifier.__class__.__name__,
            "trained": self._classifier_trained,
            "training_report": training_report,
        }

    def build_dataset(self) -> PhaseDataset:
        return dataset_from_records(self.logger.iter_experiments())

    def train_classifier(self, dataset: Optional[PhaseDataset] = None) -> Dict[str, Any]:
        dataset = dataset or self.build_dataset()
        if not dataset.examples:
            raise ValueError("Cannot train classifier with empty dataset")

        if self.classifier is self.mlp_classifier:
            if self.mlp_classifier is None or not self.mlp_classifier.is_available():
                raise RuntimeError(
                    "MLPPhaseClassifier unavailable. Install torch or switch to centroid classifier."
                )

        report = self.classifier.train(dataset)
        self._classifier_trained = True
        if self.classifier is self.centroid_classifier:
            self._centroid_classifier_trained = True
        elif self.classifier is self.mlp_classifier:
            self._mlp_classifier_trained = True
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_phase_experiment(
        self,
        *,
        phase_type: str,
        system_size: int,
        depth: int,
        seed: int,
        top_k: int,
        phase_options: Dict[str, Any],
        experiment_id: Optional[str] = None,
        register: bool = True,
    ) -> Dict[str, Any]:
        generator = self.generators[phase_type]
        config = generator(system_size, depth, seed, dict(phase_options))

        params = {
            "phase_type": phase_type,
            "system_size": system_size,
            "depth": depth,
            "seed": seed,
            "top_k": top_k,
            **phase_options,
        }

        signature = self._signature_for_run(phase_type, params)
        experiment_id = experiment_id or self._build_experiment_id(phase_type, params)

        self.logger.start_experiment(experiment_id, phase_type, params)
        hooks = self._build_hooks(experiment_id)

        payload: Dict[str, Any] = {
            "base_state": config.base_state,
            "variations": config.variations,
            "top_k": top_k,
            "priority": 2.5,
            "__hooks__": hooks,
            "__experiment_metadata__": {
                "experiment_id": experiment_id,
                "phase_type": phase_type,
                "params": params,
                **dict(config.metadata),
            },
        }
        if config.scoring_key:
            payload["scoring_key"] = config.scoring_key
        if config.scoring_weights:
            payload["scoring_weights"] = config.scoring_weights

        raw_result = self.orchestrator.submit("quantum", signature, payload)
        collapsed_state = dict(raw_result.get("collapsed_state", {}))
        summary = self._compose_summary(params, collapsed_state, raw_result)

        self.logger.log_final_state(experiment_id, collapsed_state, summary)

        record = self.logger.get_experiment(experiment_id)
        if record is None:
            raise RuntimeError(f"Experiment {experiment_id} failed to log")

        feature_vector = extract_features(record)
        self.logger.attach_feature_vector(experiment_id, feature_vector)

        if not register:
            self.registry.pop(experiment_id, None)

        return {
            "experiment_id": experiment_id,
            "signature": signature,
            "params": params,
            "feature_vector": feature_vector,
            "summary": summary,
            "collapsed_state": collapsed_state,
            "raw_result": raw_result,
        }

    def _build_hooks(self, experiment_id: str) -> QuantumExperimentHooks:
        def on_spawn(branches, metadata):
            metadata = dict(metadata)
            metadata.setdefault("experiment_id", experiment_id)
            metadata.setdefault("stage", "spawn")
            metadata.setdefault("layer_index", 0)
            self.logger.log_spawn(experiment_id, branches, metadata)

        def on_interfere(branches, metadata):
            metadata = dict(metadata)
            metadata.setdefault("experiment_id", experiment_id)
            metadata.setdefault("stage", "interfere")
            metadata.setdefault("layer_index", 1)
            self.logger.log_interfere(experiment_id, branches, metadata)

        def on_collapse(branches, metadata):
            metadata = dict(metadata)
            metadata.setdefault("experiment_id", experiment_id)
            metadata.setdefault("stage", "collapse")
            metadata.setdefault("layer_index", 2)
            self.logger.log_collapse(experiment_id, branches, metadata)

        return QuantumExperimentHooks(
            on_spawn=on_spawn,
            on_interfere=on_interfere,
            on_collapse=on_collapse,
        )

    def _signature_for_run(self, phase_type: str, params: Dict[str, Any]) -> str:
        fingerprint = self._fingerprint(params)
        return f"phase::{phase_type}::{fingerprint}"

    def _build_experiment_id(self, phase_type: str, params: Dict[str, Any]) -> str:
        fingerprint = self._fingerprint(params)
        suffix = uuid.uuid4().hex[:6]
        return f"{phase_type}::{fingerprint}::{suffix}"

    def _fingerprint(self, params: Dict[str, Any]) -> str:
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def _compose_summary(
        self,
        params: Dict[str, Any],
        collapsed_state: Dict[str, Any],
        raw_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        magnetization = collapsed_state.get("magnetization")
        if magnetization is None:
            spins = collapsed_state.get("spins")
            if isinstance(spins, list):
                magnetization = _magnetization(spins)

        summary = {
            "phase_type": params.get("phase_type"),
            "system_size": params.get("system_size"),
            "depth": params.get("depth"),
            "seed": params.get("seed"),
            "top_k": params.get("top_k"),
            "magnetization": magnetization,
            "entropy_proxy": collapsed_state.get("entropy_proxy"),
            "correlation_length": collapsed_state.get("correlation_length"),
            "string_order": collapsed_state.get("string_order"),
            "randomness_score": collapsed_state.get("randomness_score"),
            "branch_count": raw_result.get("branch_count"),
            "interfered_count": raw_result.get("interfered_count"),
        }
        summary = {k: v for k, v in summary.items() if v is not None}
        return summary


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_detector: Optional[PhaseDetector] = None


def configure_phase_detector(detector: PhaseDetector) -> None:
    global _default_detector
    _default_detector = detector


def get_phase_detector() -> PhaseDetector:
    if _default_detector is None:
        raise RuntimeError("PhaseDetector is not configured")
    return _default_detector


def run_phase_experiment(**kwargs: Any) -> Dict[str, Any]:
    detector = get_phase_detector()
    return detector.run_phase_experiment(**kwargs)


def log_phase_features(experiment_id: str) -> List[float]:
    detector = get_phase_detector()
    return detector.log_phase_features(experiment_id)


def replay_experiment(experiment_id: str, *, compare: bool = True) -> Dict[str, Any]:
    detector = get_phase_detector()
    return detector.replay_experiment(experiment_id, compare=compare)


def classify_phase(
    *,
    experiment_id: Optional[str] = None,
    feature_vector: Optional[List[float]] = None,
    retrain: bool = False,
) -> Dict[str, Any]:
    detector = get_phase_detector()
    return detector.classify_phase(
        experiment_id=experiment_id,
        feature_vector=feature_vector,
        retrain=retrain,
    )
