"""Synthetic GPU Adapter Cluster - heterogeneous compute scheduler."""

from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

from .types import Device, DeviceType, WorkUnit


@dataclass
class Batch:
    """Collection of WorkUnits assigned to a single device."""

    device_id: str
    work_units: List[WorkUnit]
    total_cost: int


class SyntheticAdapterCluster:
    """Manages heterogeneous devices and schedules work units adaptively."""

    def __init__(
        self,
        devices: List[Device],
        executor: Optional[concurrent.futures.Executor] = None,
        work_handler: Optional[Callable[[Batch], Dict[str, object]]] = None,
    ) -> None:
        """Initialize the adapter cluster.

        Args:
            devices: List of compute devices available.
            executor: Optional executor for async work. If None, creates ThreadPoolExecutor.
            work_handler: Optional custom handler for processing batches.
        """
        self.devices: Dict[str, Device] = {d.id: d for d in devices}
        self.work_queue: Deque[WorkUnit] = deque()
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._futures: Dict[str, tuple[concurrent.futures.Future, Batch, float]] = {}
        self._work_handler = work_handler or self._default_work_handler
        self._shutdown = False

    def submit_work_units(self, work_units: List[WorkUnit]) -> None:
        """Add work units to the queue."""
        self.work_queue.extend(work_units)

    def step(self) -> None:
        """Execute one scheduler step: dispatch idle devices, collect completed work."""
        self._dispatch_work()
        self._collect_results()

    def shutdown(self) -> None:
        """Shut down the cluster and executor."""
        self._shutdown = True
        self._executor.shutdown(wait=True)

    def get_stats(self) -> Dict[str, object]:
        """Retrieve cluster statistics."""
        return {
            "queue_length": len(self.work_queue),
            "active_tasks": len(self._futures),
            "devices": {
                d.id: {
                    "type": d.type.value,
                    "perf_score": d.perf_score,
                    "batch_size": d.batch_size,
                    "is_busy": d.is_busy,
                    "last_latency": d.last_latency,
                }
                for d in self.devices.values()
            },
        }

    def _dispatch_work(self) -> None:
        """Dispatch work to idle devices."""
        for device in self.devices.values():
            if device.is_busy or not self.work_queue:
                continue

            batch_size = device.batch_size
            units: List[WorkUnit] = []
            total_cost = 0
            while self.work_queue and len(units) < batch_size:
                unit = self.work_queue.popleft()
                units.append(unit)
                total_cost += unit.cost_hint

            if not units:
                continue

            batch = Batch(device_id=device.id, work_units=units, total_cost=total_cost)
            future = self._executor.submit(self._work_handler, batch)
            self._futures[device.id] = (future, batch, time.perf_counter())
            device.is_busy = True

    def _collect_results(self) -> None:
        """Collect completed work and update device metrics."""
        completed_devices: List[str] = []
        for device_id, (future, batch, start_time) in self._futures.items():
            if not future.done():
                continue

            try:
                result = future.result(timeout=0)
                elapsed = result.get("elapsed", time.perf_counter() - start_time)
            except Exception:
                elapsed = time.perf_counter() - start_time

            device = self.devices[device_id]
            device.is_busy = False
            device.last_latency = elapsed
            device.adjust_batch_size(elapsed)

            if elapsed > 0 and batch.total_cost > 0:
                throughput = batch.total_cost / elapsed
                device.perf_score = 0.9 * device.perf_score + 0.1 * throughput

            completed_devices.append(device_id)

        for device_id in completed_devices:
            del self._futures[device_id]

    def _default_work_handler(self, batch: Batch) -> Dict[str, object]:
        """Default handler simulates work based on device type and cost."""
        device = self.devices[batch.device_id]
        start = time.perf_counter()

        if device.type == DeviceType.GPU:
            speedup = 4.0
        elif device.type == DeviceType.VIRTUAL:
            speedup = 2.0
        else:
            speedup = 1.0

        base_time = batch.total_cost * 0.0001
        simulated_sleep = base_time / speedup
        time.sleep(max(simulated_sleep, 0.001))

        elapsed = time.perf_counter() - start
        return {
            "device_id": batch.device_id,
            "units_processed": len(batch.work_units),
            "total_cost": batch.total_cost,
            "elapsed": elapsed,
        }


__all__ = ["SyntheticAdapterCluster", "Batch"]
