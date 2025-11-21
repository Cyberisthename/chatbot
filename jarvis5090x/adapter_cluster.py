from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Iterable, List, Optional

from .types import AdapterAssignment, AdapterDevice, OperationKind, OperationRequest


@dataclass
class _DeviceState:
    active: int = 0
    last_activity: float = 0.0


class AdapterCluster:
    """Lightweight multi-device scheduler for heterogeneous workloads."""

    def __init__(self, devices: Iterable[AdapterDevice]) -> None:
        self._devices: Dict[str, AdapterDevice] = {device.id: device for device in devices}
        self._device_state: Dict[str, _DeviceState] = {
            device_id: _DeviceState() for device_id in self._devices
        }
        self._queue: List[OperationRequest] = []
        self._lock = Lock()

    def add_device(self, device: AdapterDevice) -> None:
        with self._lock:
            self._devices[device.id] = device
            self._device_state[device.id] = _DeviceState()

    def submit_batch(self, batch: Iterable[OperationRequest]) -> None:
        with self._lock:
            for task in batch:
                self._queue.append(task)
            self._queue.sort(key=lambda task: (-task.priority, task.op_signature))

    def step(self) -> List[AdapterAssignment]:
        with self._lock:
            assignments: List[AdapterAssignment] = []
            if not self._queue:
                return assignments

            available_devices = [
                (device, self._device_state[device.id])
                for device in self._devices.values()
                if self._device_state[device.id].active < device.max_concurrency
            ]
            available_devices.sort(key=lambda item: (-item[0].perf_score, item[1].last_activity))

            for device, state in available_devices:
                task_index = self._find_task_for_device(device)
                if task_index is None:
                    continue
                task = self._queue.pop(task_index)
                assignments.append(AdapterAssignment(device=device, task=task))
                state.active += 1
                state.last_activity = time.perf_counter()
            return assignments

    def complete(self, device_id: str) -> None:
        with self._lock:
            state = self._device_state.get(device_id)
            if state and state.active > 0:
                state.active -= 1
                state.last_activity = time.perf_counter()

    def pending_tasks(self) -> int:
        with self._lock:
            return len(self._queue)

    def _find_task_for_device(self, device: AdapterDevice) -> Optional[int]:
        for index, task in enumerate(self._queue):
            if device.supports(task.op_type):
                return index
        return None

    @property
    def devices(self) -> Dict[str, AdapterDevice]:
        return self._devices
