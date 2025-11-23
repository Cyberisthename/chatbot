from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .phase_features import extract_features
from .phase_logger import ExperimentRecord, PhaseLogger


class PhaseReplay:
    def __init__(self, logger: PhaseLogger) -> None:
        self._logger = logger

    def can_replay(self, experiment_id: str) -> bool:
        record = self._logger.get_experiment(experiment_id)
        return record is not None and record.end_time is not None

    def get_replay_config(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        record = self._logger.get_experiment(experiment_id)
        if record is None:
            return None

        return {
            "phase_type": record.phase_type,
            "params": dict(record.params),
            "experiment_id": record.experiment_id,
        }

    def compare_experiments(
        self,
        original_id: str,
        replayed_id: str,
    ) -> Dict[str, Any]:
        original = self._logger.get_experiment(original_id)
        replayed = self._logger.get_experiment(replayed_id)

        if original is None or replayed is None:
            return {"error": "Experiment not found"}

        original_features = extract_features(original)
        replayed_features = extract_features(replayed)

        if len(original_features) != len(replayed_features):
            return {"error": "Feature vector length mismatch"}

        differences = [
            abs(o - r) for o, r in zip(original_features, replayed_features)
        ]
        max_diff = max(differences) if differences else 0.0
        mean_diff = sum(differences) / len(differences) if differences else 0.0

        return {
            "original_id": original_id,
            "replayed_id": replayed_id,
            "feature_count": len(original_features),
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "is_match": max_diff < 1e-6,
            "original_features": original_features,
            "replayed_features": replayed_features,
        }

    def validate_replay(self, experiment_id: str) -> Tuple[bool, str]:
        record = self._logger.get_experiment(experiment_id)
        if record is None:
            return False, f"Experiment {experiment_id} not found"

        if record.end_time is None:
            return False, "Experiment not completed"

        if not record.layers:
            return False, "No layer logs recorded"

        if record.final_state is None:
            return False, "No final state recorded"

        return True, "Experiment is valid for replay"
