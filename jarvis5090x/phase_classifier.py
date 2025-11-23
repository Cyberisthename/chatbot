from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .phase_dataset import PhaseDataset, PhaseExample


class SimplePhaseClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._training_examples: List[PhaseExample] = []
        self._label_set: List[str] = []

    def train(self, dataset: PhaseDataset) -> Dict[str, Any]:
        self._training_examples = list(dataset.examples)
        self._label_set = sorted(
            set(example.phase_label for example in self._training_examples)
        )

        return {
            "training_examples": len(self._training_examples),
            "unique_labels": len(self._label_set),
            "labels": self._label_set,
        }

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        if not self._training_examples:
            return "unknown", 0.0

        distances: List[Tuple[float, str]] = []
        for example in self._training_examples:
            distance = self._euclidean_distance(feature_vector, example.feature_vector)
            distances.append((distance, example.phase_label))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[: min(self.k, len(distances))]

        label_counts = Counter(label for _distance, label in k_nearest)
        most_common_label, count = label_counts.most_common(1)[0]
        confidence = count / len(k_nearest)

        return most_common_label, confidence

    def evaluate(self, test_dataset: PhaseDataset) -> Dict[str, Any]:
        if not test_dataset.examples:
            return {"error": "Empty test dataset"}

        correct = 0
        total = len(test_dataset.examples)
        confusion: Dict[str, Dict[str, int]] = {
            label: {other: 0 for other in self._label_set} for label in self._label_set
        }
        confidences: List[float] = []

        for example in test_dataset.examples:
            predicted_label, confidence = self.predict(example.feature_vector)
            true_label = example.phase_label

            if predicted_label == true_label:
                correct += 1

            if true_label in confusion and predicted_label in confusion[true_label]:
                confusion[true_label][predicted_label] += 1

            confidences.append(confidence)

        accuracy = correct / total if total > 0 else 0.0
        mean_confidence = statistics.mean(confidences) if confidences else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mean_confidence": mean_confidence,
            "confusion_matrix": confusion,
        }

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class CentroidPhaseClassifier:
    def __init__(self) -> None:
        self._centroids: Dict[str, List[float]] = {}
        self._label_set: List[str] = []

    def train(self, dataset: PhaseDataset) -> Dict[str, Any]:
        if not dataset.examples:
            return {"error": "Empty dataset"}

        examples_by_label: Dict[str, List[List[float]]] = {}
        for example in dataset.examples:
            label = example.phase_label
            if label not in examples_by_label:
                examples_by_label[label] = []
            examples_by_label[label].append(example.feature_vector)

        self._centroids = {}
        for label, vectors in examples_by_label.items():
            self._centroids[label] = self._compute_centroid(vectors)

        self._label_set = sorted(self._centroids.keys())

        return {
            "training_examples": len(dataset.examples),
            "unique_labels": len(self._centroids),
            "labels": self._label_set,
        }

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        if not self._centroids:
            return "unknown", 0.0

        distances: Dict[str, float] = {}
        for label, centroid in self._centroids.items():
            distances[label] = self._euclidean_distance(feature_vector, centroid)

        min_label = min(distances, key=distances.get)  # type: ignore[arg-type]
        min_distance = distances[min_label]

        all_distances = list(distances.values())
        total_distance = sum(all_distances)
        if total_distance <= 0:
            confidence = 1.0 / len(self._centroids)
        else:
            inverse_distance = 1.0 / (min_distance + 1e-6)
            confidence = inverse_distance / sum(1.0 / (d + 1e-6) for d in all_distances)

        return min_label, confidence

    def evaluate(self, test_dataset: PhaseDataset) -> Dict[str, Any]:
        if not test_dataset.examples:
            return {"error": "Empty test dataset"}

        correct = 0
        total = len(test_dataset.examples)
        confusion: Dict[str, Dict[str, int]] = {
            label: {other: 0 for other in self._label_set} for label in self._label_set
        }
        confidences: List[float] = []

        for example in test_dataset.examples:
            predicted_label, confidence = self.predict(example.feature_vector)
            true_label = example.phase_label

            if predicted_label == true_label:
                correct += 1

            if true_label in confusion and predicted_label in confusion[true_label]:
                confusion[true_label][predicted_label] += 1

            confidences.append(confidence)

        accuracy = correct / total if total > 0 else 0.0
        mean_confidence = statistics.mean(confidences) if confidences else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mean_confidence": mean_confidence,
            "confusion_matrix": confusion,
        }

    def _compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []

        length = len(vectors[0])
        centroid = [0.0 for _ in range(length)]

        for vector in vectors:
            for idx, value in enumerate(vector):
                if idx < length:
                    centroid[idx] += value

        for idx in range(length):
            centroid[idx] /= len(vectors)

        return centroid

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
