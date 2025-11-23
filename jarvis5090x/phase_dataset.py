from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .phase_logger import ExperimentRecord


@dataclass
class PhaseExample:
    experiment_id: str
    phase_label: str
    feature_vector: List[float]
    params: Dict[str, Any]


@dataclass
class PhaseDataset:
    examples: List[PhaseExample] = field(default_factory=list)

    def add_example(self, example: PhaseExample) -> None:
        self.examples.append(example)

    def extend(self, examples: Iterable[PhaseExample]) -> None:
        self.examples.extend(examples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": [
                {
                    "experiment_id": example.experiment_id,
                    "phase_label": example.phase_label,
                    "feature_vector": example.feature_vector,
                    "params": example.params,
                }
                for example in self.examples
            ]
        }

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def __len__(self) -> int:
        return len(self.examples)

    def split(self, ratio: float = 0.8) -> Tuple["PhaseDataset", "PhaseDataset"]:
        cutoff = int(len(self.examples) * ratio)
        train = PhaseDataset(self.examples[:cutoff])
        test = PhaseDataset(self.examples[cutoff:])
        return train, test


def dataset_from_records(records: Iterable[ExperimentRecord]) -> PhaseDataset:
    dataset = PhaseDataset()
    for record in records:
        if record.feature_vector is None:
            continue
        dataset.add_example(
            PhaseExample(
                experiment_id=record.experiment_id,
                phase_label=record.phase_type,
                feature_vector=list(record.feature_vector),
                params=dict(record.params),
            )
        )
    return dataset


def merge_datasets(*datasets: PhaseDataset) -> PhaseDataset:
    merged = PhaseDataset()
    for dataset in datasets:
        merged.extend(dataset.examples)
    return merged
