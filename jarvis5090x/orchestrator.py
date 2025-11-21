from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

from .adapter_cluster import AdapterCluster
from .config import DEFAULT_CONFIG, EXTREME_CONFIG, Jarvis5090XConfig
from .flop_compression import FlopCompressionLayer
from .infinite_cache import InfiniteMemoryCache
from .quantum_layer import QuantumApproximationLayer
from .types import AdapterAssignment, AdapterDevice, DeviceKind, OperationKind, OperationRequest

try:  # pragma: no cover - optional integration
    from synthetic_gpu_miner.hash_core import HashCore
    from synthetic_gpu_miner.work_unit import Batch, DeviceType, WorkUnit
except Exception:  # pragma: no cover - hash core optional
    HashCore = None  # type: ignore
    Batch = None  # type: ignore
    DeviceType = None  # type: ignore
    WorkUnit = None  # type: ignore

try:  # pragma: no cover - optional integration
    from synthetic_gpu_miner.device_manager import DeviceManager
except Exception:  # pragma: no cover - device manager optional
    DeviceManager = None  # type: ignore


class Jarvis5090X:
    """Unified orchestrator that powers the Jarvis-5090X virtual GPU."""

    @classmethod
    def build_extreme(
        cls,
        devices: List[AdapterDevice],
        **kwargs: Any,
    ) -> "Jarvis5090X":
        """Create an orchestrator using the EXTREME_CONFIG profile."""
        kwargs.pop("config", None)
        return cls(devices, config=EXTREME_CONFIG, **kwargs)

    def __init__(
        self,
        devices: List[AdapterDevice],
        compression_layer: Optional[FlopCompressionLayer] = None,
        cache_layer: Optional[InfiniteMemoryCache] = None,
        quantum_layer: Optional[QuantumApproximationLayer] = None,
        adapter_cluster: Optional[AdapterCluster] = None,
        config: Jarvis5090XConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        compression = compression_layer or FlopCompressionLayer(
            max_bases=config.compression_max_bases,
            stability_threshold=config.compression_stability_threshold,
            tolerance=config.compression_tolerance,
        )
        cache = cache_layer or InfiniteMemoryCache(max_items=config.cache_max_items)
        quantum = quantum_layer or QuantumApproximationLayer(
            max_branches=config.quantum_max_branches,
            seed=config.quantum_seed,
        )

        self.compression = compression
        self.cache = cache
        self.quantum = quantum
        self.cluster = adapter_cluster or AdapterCluster(devices)
        self.devices: Dict[str, AdapterDevice] = {device.id: device for device in devices}
        self._bootstrap_devices_from_manager()

        self._hash_core = HashCore() if HashCore is not None else None

        self._stats: Dict[str, Any] = {
            "total_ops": 0,
            "cache_hits": 0,
            "backend_executions": 0,
            "compressions": 0,
            "total_latency": 0.0,
            "backends": {kind.value: 0 for kind in OperationKind},
        }

    def submit(self, op_type: str, op_signature: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a workload to the Jarvis-5090X orchestrator."""
        start = time.perf_counter()
        self._stats["total_ops"] += 1

        op_kind = OperationKind.from_value(op_type)
        cache_key = f"{op_kind.value}:{op_signature}"

        cached = self.cache.lookup(cache_key, payload)
        if cached is not None:
            self._stats["cache_hits"] += 1
            self._stats["total_latency"] += time.perf_counter() - start
            return cached

        compressed_payload = self.compression.maybe_compress(op_type, op_signature, payload)
        if compressed_payload.get("__compressed__"):
            self._stats["compressions"] += 1

        request = OperationRequest(
            op_type=op_kind,
            op_signature=cache_key,
            payload=compressed_payload,
            priority=self._determine_priority(op_kind, compressed_payload),
        )

        assignment = self._reserve_assignment(request)
        device = assignment.device if assignment else self._fallback_device(op_kind)

        result = self._execute_backend(op_kind, compressed_payload, device)
        elapsed = time.perf_counter() - start
        self._stats["total_latency"] += elapsed
        self._stats["backend_executions"] += 1
        self._stats["backends"][op_kind.value] += 1

        if assignment:
            self.cluster.complete(assignment.device.id)

        self.cache.store(cache_key, payload, result)
        return result

    def benchmark_stats(self) -> Dict[str, Any]:
        cache_stats = self.cache.stats()
        compression_stats = self.compression.stats()
        total_ops = max(self._stats["total_ops"], 1)
        avg_latency = (self._stats["total_latency"] / total_ops) * 1000
        hit_rate = cache_stats["hit_rate_pct"]
        estimated_tflops = self._estimate_tflops(cache_stats, compression_stats)

        return {
            "total_ops": self._stats["total_ops"],
            "cache_hits": cache_stats["hits"],
            "compressions": self._stats["compressions"],
            "backend_executions": self._stats["backend_executions"],
            "average_latency_ms": round(avg_latency, 4),
            "backends": dict(self._stats["backends"]),
            "cache": cache_stats,
            "compression": compression_stats,
            "estimated_tflops": estimated_tflops,
            "effective_speedup_pct": round(hit_rate * 0.5 + compression_stats.get("stable_bases", 0) * 2, 2),
        }

    def shutdown(self) -> None:
        if self._hash_core is not None and hasattr(self._hash_core, "shutdown"):
            self._hash_core.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_devices_from_manager(self) -> None:
        if DeviceManager is None:
            return
        try:
            manager = DeviceManager()
            for device in manager.get_all_devices():
                adapter_id = f"dm::{device.id}"
                if adapter_id in self.devices:
                    continue
                kind = DeviceKind.GPU if getattr(device, "type", None).value == "GPU" else DeviceKind.CPU
                capabilities = {OperationKind.HASHING, OperationKind.LINALG, OperationKind.GENERIC}
                adapter = AdapterDevice(
                    id=adapter_id,
                    label=device.metadata.get("name", device.id) if hasattr(device, "metadata") else device.id,
                    kind=kind,
                    perf_score=getattr(device, "perf_score", 1.0),
                    max_concurrency=max(1, int(getattr(device, "perf_score", 1.0) // 100) + 1),
                    capabilities=capabilities,
                    metadata=getattr(device, "metadata", {}),
                )
                self.devices[adapter.id] = adapter
                self.cluster.add_device(adapter)
        except Exception:
            # Device detection is best-effort; ignore failures in limited environments
            pass

    def _reserve_assignment(self, request: OperationRequest) -> Optional[AdapterAssignment]:
        self.cluster.submit_batch([request])
        assignments = self.cluster.step()
        target: Optional[AdapterAssignment] = None
        for assignment in assignments:
            if assignment.task.request_id == request.request_id:
                target = assignment
            else:
                # Return other tasks to the queue and free the device immediately
                self.cluster.submit_batch([assignment.task])
                self.cluster.complete(assignment.device.id)
        return target

    def _fallback_device(self, op_kind: OperationKind) -> Optional[AdapterDevice]:
        capable_devices = [device for device in self.devices.values() if device.supports(op_kind)]
        if not capable_devices:
            return None
        capable_devices.sort(key=lambda device: device.perf_score, reverse=True)
        return capable_devices[0]

    def _execute_backend(
        self,
        op_kind: OperationKind,
        payload: Dict[str, Any],
        device: Optional[AdapterDevice],
    ) -> Dict[str, Any]:
        if op_kind == OperationKind.HASHING:
            return self._execute_hashing(payload, device)
        if op_kind == OperationKind.QUANTUM:
            return self._execute_quantum(payload)
        if op_kind == OperationKind.LINALG:
            return self._execute_linalg(payload, device)
        return self._execute_generic(payload)

    def _execute_hashing(self, payload: Dict[str, Any], device: Optional[AdapterDevice]) -> Dict[str, Any]:
        if self._hash_core is None or Batch is None or WorkUnit is None or DeviceType is None:
            data = payload.get("header_prefix") or payload.get("data") or str(payload)
            if isinstance(data, bytes):
                raw = data
            elif isinstance(data, str):
                try:
                    raw = bytes.fromhex(data)
                except ValueError:
                    raw = data.encode("utf-8")
            else:
                raw = str(data).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            return {"hash": digest, "algorithm": "sha256"}

        header_prefix = self._normalize_header_prefix(payload.get("header_prefix"))
        nonce_start = int(payload.get("nonce_start", 0))
        nonce_count = max(1, int(payload.get("nonce_count", 1)))
        target = int(payload.get("target", (1 << 224) - 1))

        work_unit = WorkUnit(
            job_id=payload.get("job_id", "jarvis5090x_job"),
            midstate_id=payload.get("midstate_id", "midstate_default"),
            nonce_start=nonce_start,
            nonce_count=nonce_count,
        )
        batch = Batch.from_work_units(
            device_id=device.id if device else "jarvis5090x_virtual",
            work_units=[work_unit],
        )
        midstate_payload = {"header_prefix": header_prefix}
        device_type = DeviceType.GPU if device and device.kind == DeviceKind.GPU else DeviceType.CPU
        future = self._hash_core.submit_batch(batch, midstate_payload, target, device_type)
        core_output = future.result()
        shares = core_output.get("shares_found", []) if isinstance(core_output, dict) else []
        if shares:
            hash_hex = shares[0].get("hash")
        else:
            nonce_bytes = nonce_start.to_bytes(4, "little", signed=False)
            hash_hex = hashlib.sha256(header_prefix + nonce_bytes).hexdigest()
        return {
            "hash": hash_hex,
            "hashes_processed": core_output.get("hashes_processed", nonce_count) if isinstance(core_output, dict) else nonce_count,
            "shares_found": shares,
            "device_id": core_output.get("device_id") if isinstance(core_output, dict) else (device.id if device else None),
            "elapsed": core_output.get("elapsed") if isinstance(core_output, dict) else None,
            "raw_result": core_output,
        }

    def _execute_quantum(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        base_state = copy.deepcopy(payload.get("base_state", {}))
        variations = copy.deepcopy(payload.get("variations", []))
        top_k = max(1, int(payload.get("top_k", 1)))
        scoring_fn = self._build_scoring_fn(payload)

        branches = self.quantum.spawn(base_state, variations)
        interfered = self.quantum.interfere(branches, scoring_fn)
        collapsed = self.quantum.collapse(interfered, top_k=top_k)

        return {
            "branch_count": len(branches),
            "collapsed_state": collapsed,
            "result": collapsed,
        }

    def _execute_linalg(self, payload: Dict[str, Any], device: Optional[AdapterDevice]) -> Dict[str, Any]:
        operation = payload.get("operation", "matmul").lower()
        matrix = copy.deepcopy(payload.get("matrix", []))
        vector = copy.deepcopy(payload.get("vector", []))
        matrix_b = copy.deepcopy(payload.get("matrix_b"))

        if operation == "matmul":
            if matrix_b is not None:
                return {"result": self._matrix_multiply(matrix, matrix_b), "operation": operation}
            return {"result": self._matrix_vector_multiply(matrix, vector), "operation": operation}
        if operation == "dot":
            return {"result": self._dot_product(matrix, vector), "operation": operation}
        if operation == "norm":
            return {"result": self._vector_norm(vector), "operation": operation}
        return {"result": payload, "operation": operation}

    def _execute_generic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": copy.deepcopy(payload), "passthrough": True}

    def _determine_priority(self, op_kind: OperationKind, payload: Dict[str, Any]) -> float:
        if "priority" in payload:
            try:
                return float(payload["priority"])
            except (TypeError, ValueError):
                pass
        priority_map = {
            OperationKind.HASHING: 3.0,
            OperationKind.QUANTUM: 2.0,
            OperationKind.LINALG: 1.5,
            OperationKind.GENERIC: 1.0,
        }
        return priority_map.get(op_kind, 1.0)

    def _build_scoring_fn(self, payload: Dict[str, Any]) -> Callable[[Dict[str, Any]], float]:
        scoring_fn = payload.get("scoring_fn")
        if callable(scoring_fn):
            return scoring_fn  # type: ignore[return-value]
        scoring_key = payload.get("scoring_key")
        if isinstance(scoring_key, str):
            return lambda state: float(state.get(scoring_key, 0.0))
        scoring_weights = payload.get("scoring_weights")
        if isinstance(scoring_weights, dict):
            weights: Dict[str, float] = {}
            for key, value in scoring_weights.items():
                try:
                    weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if weights:
                return lambda state: sum(float(state.get(k, 0.0)) * w for k, w in weights.items())
        return lambda _state: 1.0

    def _normalize_header_prefix(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("0x"):
                stripped = stripped[2:]
            try:
                return bytes.fromhex(stripped)
            except ValueError:
                return stripped.encode("utf-8")
        if value is None:
            return b""
        return str(value).encode("utf-8")

    def _matrix_vector_multiply(self, matrix: Any, vector: Any) -> List[float]:
        if not matrix or not vector:
            return []
        result: List[float] = []
        for row in matrix:
            row_vals = list(row) if isinstance(row, (list, tuple)) else [row]
            if len(row_vals) != len(vector):
                raise ValueError("Matrix row length must match vector length")
            result.append(sum(float(a) * float(b) for a, b in zip(row_vals, vector)))
        return result

    def _matrix_multiply(self, a: Any, b: Any) -> List[List[float]]:
        if not a or not b:
            return []
        rows_a = len(a)
        cols_a = len(a[0]) if isinstance(a[0], (list, tuple)) else 1
        cols_b = len(b[0]) if isinstance(b[0], (list, tuple)) else len(b)
        result: List[List[float]] = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        for i, row in enumerate(a):
            row_vals = list(row) if isinstance(row, (list, tuple)) else [row]
            for j in range(cols_b):
                col_vals = [row_b[j] for row_b in b]
                result[i][j] = sum(float(x) * float(y) for x, y in zip(row_vals, col_vals))
        return result

    def _dot_product(self, a: Any, b: Any) -> float:
        if not a or not b:
            return 0.0
        return sum(float(x) * float(y) for x, y in zip(a, b))

    def _vector_norm(self, vector: Any) -> float:
        return sum(float(x) ** 2 for x in vector) ** 0.5

    def _estimate_tflops(
        self,
        cache_stats: Dict[str, Any],
        compression_stats: Dict[str, Any],
    ) -> float:
        base_tflops = 125.0  # RTX 5090 reference
        cache_multiplier = 1.0 + (cache_stats.get("hit_rate_pct", 0.0) / 100.0) * 0.4
        stable_bases = compression_stats.get("stable_bases", 0)
        basis_count = max(compression_stats.get("basis_count", 1), 1)
        compression_multiplier = 1.0 + (stable_bases / basis_count) * 0.6
        return round(base_tflops * cache_multiplier * compression_multiplier, 2)
