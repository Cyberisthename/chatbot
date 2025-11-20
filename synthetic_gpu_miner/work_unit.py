from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class DeviceType(enum.Enum):
    GPU = "GPU"
    CPU = "CPU"


@dataclass(slots=True)
class WorkUnit:
    job_id: str
    midstate_id: str
    nonce_start: int
    nonce_count: int

    def split(self, max_count: int) -> List['WorkUnit']:
        if max_count <= 0 or self.nonce_count <= max_count:
            return [self]
        remaining = self.nonce_count
        current_start = self.nonce_start
        parts: List[WorkUnit] = []
        while remaining > 0:
            take = min(max_count, remaining)
            parts.append(
                WorkUnit(
                    job_id=self.job_id,
                    midstate_id=self.midstate_id,
                    nonce_start=current_start,
                    nonce_count=take,
                )
            )
            current_start += take
            remaining -= take
        return parts


@dataclass(slots=True)
class Device:
    id: str
    type: DeviceType
    perf_score: float = 1.0
    batch_size: int = 8192
    max_batch_size: int = 131072
    min_batch_size: int = 1024
    last_latency: float = 0.0
    error_rate: float = 0.0
    is_busy: bool = False
    metadata: dict = field(default_factory=dict)

    def adjust_batch_size(self, target_latency: float) -> None:
        if self.last_latency <= 0 or target_latency <= 0:
            return
        ratio = target_latency / self.last_latency
        if ratio > 1.1:
            self.batch_size = min(int(self.batch_size * 1.5), self.max_batch_size)
        elif ratio < 0.9:
            self.batch_size = max(int(self.batch_size * 0.7), self.min_batch_size)


@dataclass(slots=True)
class Batch:
    device_id: str
    work_units: List[WorkUnit]
    total_nonce_count: int
    job_id: str
    midstate_id: str
    metadata: Optional[dict] = None

    @classmethod
    def from_work_units(cls, device_id: str, work_units: List[WorkUnit]) -> 'Batch':
        if not work_units:
            raise ValueError("work_units must not be empty")
        job_ids = {w.job_id for w in work_units}
        if len(job_ids) != 1:
            raise ValueError("All work units in a batch must share the same job_id")
        midstate_ids = {w.midstate_id for w in work_units}
        if len(midstate_ids) != 1:
            raise ValueError("All work units in a batch must share the same midstate_id")
        total_nonce_count = sum(w.nonce_count for w in work_units)
        return cls(
            device_id=device_id,
            work_units=work_units,
            total_nonce_count=total_nonce_count,
            job_id=work_units[0].job_id,
            midstate_id=work_units[0].midstate_id,
            metadata=None,
        )
