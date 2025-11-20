import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .work_unit import Device


@dataclass
class PerformanceSnapshot:
    device_id: str
    timestamp: float
    hashes_processed: int
    elapsed: float
    shares_found: int
    error_count: int = 0

    def hashrate(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.hashes_processed / self.elapsed


class TelemetryController:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.snapshots: Dict[str, Deque[PerformanceSnapshot]] = defaultdict(lambda: deque(maxlen=window_size))
        self.device_stats: Dict[str, Dict[str, float]] = {}
        self.global_stats: Dict[str, float] = {
            'total_hashes': 0,
            'total_shares': 0,
            'total_errors': 0,
            'start_time': time.time(),
        }
        self._lock = None

    def record_result(self, result: Dict[str, object]) -> None:
        device_id = result.get('device_id', 'unknown')
        snapshot = PerformanceSnapshot(
            device_id=device_id,
            timestamp=time.time(),
            hashes_processed=result.get('hashes_processed', 0),
            elapsed=result.get('elapsed', 0.0),
            shares_found=len(result.get('shares_found', [])),
            error_count=result.get('error_count', 0),
        )
        self.snapshots[device_id].append(snapshot)
        self.global_stats['total_hashes'] += snapshot.hashes_processed
        self.global_stats['total_shares'] += snapshot.shares_found
        self.global_stats['total_errors'] += snapshot.error_count

    def get_device_avg_hashrate(self, device_id: str) -> float:
        snaps = list(self.snapshots.get(device_id, []))
        if not snaps:
            return 0.0
        total_hashes = sum(s.hashes_processed for s in snaps)
        total_elapsed = sum(s.elapsed for s in snaps)
        if total_elapsed <= 0:
            return 0.0
        return total_hashes / total_elapsed

    def get_device_avg_latency(self, device_id: str) -> float:
        snaps = list(self.snapshots.get(device_id, []))
        if not snaps:
            return 0.0
        return sum(s.elapsed for s in snaps) / len(snaps)

    def get_device_error_rate(self, device_id: str) -> float:
        snaps = list(self.snapshots.get(device_id, []))
        if not snaps:
            return 0.0
        total_errors = sum(s.error_count for s in snaps)
        return total_errors / len(snaps)

    def update_device_profiles(self, devices: List[Device]) -> None:
        for device in devices:
            avg_hashrate = self.get_device_avg_hashrate(device.id)
            avg_latency = self.get_device_avg_latency(device.id)
            error_rate = self.get_device_error_rate(device.id)
            if avg_hashrate > 0:
                device.perf_score = avg_hashrate / 1000000
            if avg_latency > 0:
                device.last_latency = avg_latency
            device.error_rate = error_rate

    def estimate_batch_size(self, device: Device, target_latency: float = 1.0) -> int:
        avg_hashrate = self.get_device_avg_hashrate(device.id)
        if avg_hashrate <= 0:
            return device.batch_size
        estimated_size = int(avg_hashrate * target_latency)
        estimated_size = max(device.min_batch_size, min(estimated_size, device.max_batch_size))
        return estimated_size

    def get_global_hashrate(self) -> float:
        elapsed = time.time() - self.global_stats['start_time']
        if elapsed <= 0:
            return 0.0
        return self.global_stats['total_hashes'] / elapsed

    def get_summary(self) -> Dict[str, object]:
        elapsed = time.time() - self.global_stats['start_time']
        return {
            'total_hashes': self.global_stats['total_hashes'],
            'total_shares': self.global_stats['total_shares'],
            'total_errors': self.global_stats['total_errors'],
            'uptime': elapsed,
            'global_hashrate': self.get_global_hashrate(),
        }

    def get_device_summary(self, device_id: str) -> Dict[str, object]:
        return {
            'device_id': device_id,
            'avg_hashrate': self.get_device_avg_hashrate(device_id),
            'avg_latency': self.get_device_avg_latency(device_id),
            'error_rate': self.get_device_error_rate(device_id),
            'snapshot_count': len(self.snapshots.get(device_id, [])),
        }

    def get_all_device_summaries(self) -> List[Dict[str, object]]:
        return [self.get_device_summary(device_id) for device_id in self.snapshots.keys()]
