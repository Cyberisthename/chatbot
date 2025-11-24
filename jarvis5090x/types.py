from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set


class DeviceKind(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    VIRTUAL = "VIRTUAL"


class OperationKind(str, Enum):
    HASHING = "hashing"
    QUANTUM = "quantum"
    LINALG = "linalg"
    GENERIC = "generic"

    @classmethod
    def from_value(cls, value: str) -> "OperationKind":
        normalized = (value or "generic").strip().lower()
        if normalized.startswith("hash"):
            return cls.HASHING
        if normalized.startswith("quant"):
            return cls.QUANTUM
        if normalized.startswith("lin") or normalized.startswith("mat"):
            return cls.LINALG
        return cls.GENERIC


@dataclass(slots=True)
class AdapterDevice:
    id: str
    label: str
    kind: DeviceKind
    perf_score: float = 1.0
    max_concurrency: int = 1
    capabilities: Set[OperationKind] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports(self, op_type: OperationKind) -> bool:
        return not self.capabilities or op_type in self.capabilities


@dataclass(slots=True)
class OperationRequest:
    op_type: OperationKind
    op_signature: str
    payload: Dict[str, Any]
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(slots=True)
class AdapterAssignment:
    device: AdapterDevice
    task: OperationRequest


__all__: List[str] = [
    "DeviceKind",
    "OperationKind",
    "AdapterDevice",
    "OperationRequest",
    "AdapterAssignment",
]
