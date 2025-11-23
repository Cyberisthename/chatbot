from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

from .phase_dataset import PhaseDataset

try:  # pragma: no cover - optional dependency import
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:  # pragma: no cover - handled gracefully at runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


if nn is not None:

    class PhaseMLP(nn.Module):
        def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)

else:  # pragma: no cover - executed only when PyTorch is unavailable

    class PhaseMLP:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required to instantiate PhaseMLP. Install torch to enable the MLP classifier."
            )


class MLPPhaseClassifier:
    def __init__(self) -> None:
        self._torch_available = torch is not None
        self.model: PhaseMLP | None = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: List[str] = []

    def is_available(self) -> bool:
        """Return True if PyTorch is available for training/evaluation."""

        return self._torch_available

    def _ensure_torch(self) -> None:
        if not self._torch_available:
            raise RuntimeError(
                "PyTorch is required for MLPPhaseClassifier. Install torch to enable this classifier."
            )

    def _build_label_maps(self, dataset: PhaseDataset) -> None:
        labels = sorted({ex.phase_label for ex in dataset.examples})
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = labels

    def train(self, dataset: PhaseDataset, epochs: int = 30, lr: float = 1e-3) -> Dict[str, Any]:
        self._ensure_torch()

        if not dataset.examples:
            return {"error": "Empty dataset"}

        self._build_label_maps(dataset)
        input_dim = len(dataset.examples[0].feature_vector)
        num_classes = len(self.label_to_idx)

        self.model = PhaseMLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)  # type: ignore[union-attr]
        criterion = nn.CrossEntropyLoss()  # type: ignore[operator]

        X = torch.tensor([ex.feature_vector for ex in dataset.examples], dtype=torch.float32)  # type: ignore[attr-defined]
        y = torch.tensor([self.label_to_idx[ex.phase_label] for ex in dataset.examples], dtype=torch.long)  # type: ignore[attr-defined]

        for _epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X)  # type: ignore[call-arg]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        return {
            "epochs": epochs,
            "labels": self.idx_to_label,
            "training_examples": len(dataset.examples),
            "unique_labels": len(self.label_to_idx),
        }

    def predict(self, feature_vector: List[float]) -> Tuple[str, float]:
        if self.model is None:
            return "unknown", 0.0

        self._ensure_torch()

        x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            logits = self.model(x)  # type: ignore[call-arg]
            probs = torch.softmax(logits, dim=-1)[0]  # type: ignore[attr-defined]
            idx = int(torch.argmax(probs).item())  # type: ignore[attr-defined]
            label = self.idx_to_label[idx]
            confidence = float(probs[idx].item())
        return label, confidence

    def evaluate(self, test_dataset: PhaseDataset) -> Dict[str, Any]:
        if not test_dataset.examples:
            return {"error": "Empty test dataset"}

        if self.model is None:
            return {"error": "Model is not trained"}

        self._ensure_torch()

        correct = 0
        total = len(test_dataset.examples)
        confusion: Dict[str, Dict[str, int]] = {
            label: {other: 0 for other in self.idx_to_label} for label in self.idx_to_label
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
