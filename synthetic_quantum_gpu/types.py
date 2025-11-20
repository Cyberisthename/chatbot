"""Shared type definitions for the Synthetic Quantum GPU stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np


@dataclass(slots=True)
class BranchState:
    """Represents a single quantum-like branch with amplitude and payload."""

    id: str
    amplitude: complex
    payload: Dict[str, Any]


@dataclass(slots=True)
class InterferenceResult:
    """Result of the interference step containing normalized branches."""

    branches: List[BranchState]
    normalized: bool


class DeviceType(str, Enum):
    """Available device types for the adapter cluster."""

    CPU = "CPU"
    GPU = "GPU"
    VIRTUAL = "VIRTUAL"


@dataclass(slots=True)
class Device:
    """Represents a compute device within the adapter cluster."""

    id: str
    type: DeviceType
    perf_score: float = 1.0
    batch_size: int = 32
    max_batch_size: int = 1024
    min_batch_size: int = 4
    is_busy: bool = False
    last_latency: float = 0.0

    def adjust_batch_size(self, measured_latency: float, target_latency: float = 0.05) -> None:
        """Adapt batch size based on observed latency."""

        if measured_latency <= 0:
            return
        ratio = target_latency / measured_latency
        if ratio > 1.1:
            self.batch_size = min(int(self.batch_size * 1.5), self.max_batch_size)
        elif ratio < 0.9:
            self.batch_size = max(int(self.batch_size * 0.7), self.min_batch_size)


@dataclass(slots=True)
class WorkUnit:
    """Minimal unit of work processed by the adapter cluster."""

    job_id: str
    payload: Dict[str, Any]
    cost_hint: int = 1


@dataclass(slots=True)
class MemoryEntry:
    """An entry stored within the infinite memory router."""

    key: str
    embedding: np.ndarray
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


__all__ = [
    "BranchState",
    "InterferenceResult",
    "DeviceType",
    "Device",
    "WorkUnit",
    "MemoryEntry",
]
