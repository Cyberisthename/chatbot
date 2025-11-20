import hashlib
import struct
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback when numpy is unavailable
    np = None

from .work_unit import Batch, DeviceType


class HashResult:
    __slots__ = ("nonce", "hash_hex", "difficulty")

    def __init__(self, nonce: int, hash_hex: str, difficulty: float):
        self.nonce = nonce
        self.hash_hex = hash_hex
        self.difficulty = difficulty

    def to_dict(self) -> Dict[str, object]:
        return {"nonce": self.nonce, "hash": self.hash_hex, "difficulty": self.difficulty}


class HashCore:
    def __init__(self, max_workers: Optional[int] = None):
        self.cpu_pool = ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self) -> None:
        self.cpu_pool.shutdown(wait=True)

    def submit_batch(self, batch: Batch, midstate_payload: Dict[str, bytes], target: int,
                     device_type: DeviceType) -> Future:
        if device_type == DeviceType.CPU:
            return self.cpu_pool.submit(self._execute_cpu_batch, batch, midstate_payload, target)
        return self.cpu_pool.submit(self._execute_gpu_simulated_batch, batch, midstate_payload, target)

    def _execute_cpu_batch(self, batch: Batch, midstate_payload: Dict[str, bytes], target: int) -> Dict[str, object]:
        start = time.perf_counter()
        base_header = midstate_payload["header_prefix"]
        results: List[HashResult] = []
        hashes_checked = 0
        for work_unit in batch.work_units:
            hashes, shares = self._mine_range(base_header, work_unit.nonce_start, work_unit.nonce_count, target)
            hashes_checked += hashes
            results.extend(shares)
        elapsed = time.perf_counter() - start
        return {
            "device_id": batch.device_id,
            "job_id": batch.job_id,
            "midstate_id": batch.midstate_id,
            "hashes_processed": hashes_checked,
            "shares_found": [r.to_dict() for r in results],
            "elapsed": elapsed,
        }

    def _execute_gpu_simulated_batch(self, batch: Batch, midstate_payload: Dict[str, bytes], target: int) -> Dict[str, object]:
        simulated_speedup = 4.0
        start = time.perf_counter()
        cpu_result = self._execute_cpu_batch(batch, midstate_payload, target)
        cpu_elapsed = cpu_result["elapsed"]
        adjusted_elapsed = max(cpu_elapsed / simulated_speedup, cpu_elapsed * 0.25)
        cpu_result["elapsed"] = adjusted_elapsed
        return cpu_result

    def _mine_range(self, header_prefix: bytes, nonce_start: int, nonce_count: int, target: int) -> Tuple[int, List[HashResult]]:
        shares: List[HashResult] = []
        total_hashes = nonce_count
        if np is not None:
            nonces = np.arange(nonce_start, nonce_start + nonce_count, dtype=np.uint32)
        else:
            nonces = range(nonce_start, nonce_start + nonce_count)
        for nonce in nonces:
            header = header_prefix + struct.pack('<I', int(nonce))
            first = hashlib.sha256(header).digest()
            final_hash = hashlib.sha256(first).digest()
            hash_int = int.from_bytes(final_hash, byteorder='big')
            if hash_int <= target:
                difficulty = self._estimate_difficulty(hash_int, target)
                shares.append(HashResult(int(nonce), final_hash[::-1].hex(), difficulty))
        return total_hashes, shares

    def _estimate_difficulty(self, hash_int: int, target: int) -> float:
        if hash_int == 0:
            return float('inf')
        return target / hash_int
