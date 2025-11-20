import threading
import time
from collections import deque
from queue import Queue, Empty
from typing import Deque, Dict, List, Optional

from .hash_core import HashCore
from .precompute_cache import PrecomputeCache
from .protocol_layer import MiningJob, ProtocolLayer
from .telemetry import TelemetryController
from .work_unit import Batch, Device, DeviceType, WorkUnit


class SyntheticGPUScheduler:
    def __init__(self, protocol: ProtocolLayer, hash_core: HashCore,
                 precompute: PrecomputeCache, telemetry: TelemetryController,
                 devices: List[Device]):
        self.protocol = protocol
        self.hash_core = hash_core
        self.precompute = precompute
        self.telemetry = telemetry
        self.devices = {d.id: d for d in devices}
        self.work_queue: Deque[WorkUnit] = deque()
        self.current_job: Optional[MiningJob] = None
        self.midstate_payload: Optional[Dict[str, bytes]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._result_thread = threading.Thread(target=self._result_collector_loop, daemon=True)
        self._futures: Dict[str, Queue] = {}
        self.protocol.register_job_callback(self._on_new_job)
        self._scheduler_thread.start()
        self._result_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._scheduler_thread.join(timeout=2)
        self._result_thread.join(timeout=2)
        self.hash_core.shutdown()
        self.protocol.shutdown()

    def _on_new_job(self, job: MiningJob) -> None:
        with self._lock:
            self.current_job = job
            midstate_id = self.precompute.compute_midstate(job.job_id, job.header_prefix)
            self.midstate_payload = self.precompute.get_midstate_payload(midstate_id)
            self.precompute.clear_old_midstates(job.job_id)
            self.work_queue.clear()
            chunk_size = 1 << 16
            for nonce_start in range(job.nonce_start, job.nonce_end, chunk_size):
                remaining = job.nonce_end - nonce_start
                count = min(chunk_size, remaining)
                if count <= 0:
                    break
                unit = WorkUnit(
                    job_id=job.job_id,
                    midstate_id=midstate_id,
                    nonce_start=nonce_start,
                    nonce_count=count,
                )
                self.work_queue.append(unit)
            for device in self.devices.values():
                device.is_busy = False

    def _schedule_loop(self) -> None:
        while not self._stop_event.is_set():
            device = self._get_next_idle_device()
            if not device:
                time.sleep(0.01)
                continue
            work_units = self._dequeue_work(device.batch_size)
            if not work_units:
                time.sleep(0.01)
                device.is_busy = False
                continue
            batch = Batch.from_work_units(device.id, work_units)
            midstate_payload = self.midstate_payload
            if not midstate_payload:
                device.is_busy = False
                continue
            future = self.hash_core.submit_batch(
                batch=batch,
                midstate_payload=midstate_payload,
                target=self.current_job.target if self.current_job else 0,
                device_type=device.type,
            )
            result_queue = self._futures.setdefault(device.id, Queue())
            result_queue.put((future, batch))
            device.is_busy = True

    def _result_collector_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.01)
            for device_id, queue in self._futures.items():
                try:
                    future, batch = queue.get_nowait()
                except Empty:
                    continue
                if future.done():
                    self._handle_completed_future(device_id, future, batch)
                else:
                    queue.put((future, batch))

    def _handle_completed_future(self, device_id: str, future, batch: Batch) -> None:
        try:
            result = future.result()
        except Exception as exc:
            result = {
                'device_id': device_id,
                'job_id': batch.job_id,
                'midstate_id': batch.midstate_id,
                'hashes_processed': batch.total_nonce_count,
                'shares_found': [],
                'elapsed': 0.0,
                'error_count': 1,
                'error': str(exc),
            }
        self.telemetry.record_result(result)
        device = self.devices.get(device_id)
        if device:
            device.is_busy = False
            device.last_latency = result.get('elapsed', 0.0)
            if result.get('error_count', 0) == 0:
                device.batch_size = self.telemetry.estimate_batch_size(device)
        for share in result.get('shares_found', []):
            self.protocol.simulate_pool_submission({
                'job_id': batch.job_id,
                'nonce': share['nonce'],
                'hash': share['hash'],
                'difficulty': share['difficulty'],
                'device_id': device_id,
            })

    def _get_next_idle_device(self) -> Optional[Device]:
        with self._lock:
            idle_devices = [d for d in self.devices.values() if not d.is_busy]
            if not idle_devices:
                return None
            idle_devices.sort(key=lambda d: (-d.perf_score, d.last_latency))
            return idle_devices[0]

    def _dequeue_work(self, target_nonce_count: int) -> List[WorkUnit]:
        with self._lock:
            if not self.work_queue:
                return []
            collected: List[WorkUnit] = []
            total = 0
            while self.work_queue and total < target_nonce_count:
                unit = self.work_queue.popleft()
                collected.append(unit)
                total += unit.nonce_count
            return collected
