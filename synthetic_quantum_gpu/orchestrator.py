"""Orchestrator - wires all four layers into a unified Synthetic Quantum GPU."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np

from .adapter_cluster import SyntheticAdapterCluster
from .config import DEFAULT_DIM, DEFAULT_MAX_CACHE_ITEMS, DEFAULT_MAX_MEMORY_ENTRIES, DEFAULT_SEED
from .flop_compression import FlopCompressor
from .infinite_router import InfiniteMemoryRouter
from .quantum_approx import QuantumApproximator
from .types import BranchState, Device, DeviceType, WorkUnit


class SyntheticQuantumGPU:
    """Unified interface to the entire synthetic quantum GPU stack."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        max_memory_entries: int = DEFAULT_MAX_MEMORY_ENTRIES,
        max_cache_items: int = DEFAULT_MAX_CACHE_ITEMS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize the synthetic quantum GPU.

        Args:
            dim: Embedding dimension for memory router.
            max_memory_entries: Maximum entries in infinite router.
            max_cache_items: Maximum items in FLOP cache.
            seed: Random seed for determinism.
        """
        self.dim = dim
        self.seed = seed

        self.flop_compressor = FlopCompressor(max_cache_items=max_cache_items)
        self.quantum_approx = QuantumApproximator(seed=seed)
        self.memory_router = InfiniteMemoryRouter(dim=dim, max_entries=max_memory_entries)

        default_devices = [
            Device(id="cpu_0", type=DeviceType.CPU, perf_score=10.0, batch_size=16),
            Device(id="virtual_0", type=DeviceType.VIRTUAL, perf_score=20.0, batch_size=32),
        ]
        self.adapter_cluster = SyntheticAdapterCluster(devices=default_devices)

        self._task_counter = 0

    def submit_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a task to the synthetic quantum GPU.

        Supported task kinds:
            - "linear_op": Apply compressed matrix to input vector
            - "branch_and_interfere": Quantum-style branching and interference
            - "cached_function": Cache expensive computation result
        """
        kind = task.get("kind", "")
        task_id = task.get("id", f"task_{self._task_counter}")
        self._task_counter += 1

        if kind == "linear_op":
            return self._handle_linear_op(task_id, task)
        elif kind == "branch_and_interfere":
            return self._handle_branch_and_interfere(task_id, task)
        elif kind == "cached_function":
            return self._handle_cached_function(task_id, task)
        else:
            return {"id": task_id, "status": "error", "message": f"Unknown task kind: {kind}"}

    def shutdown(self) -> None:
        """Shut down all components."""
        self.adapter_cluster.shutdown()

    def _handle_linear_op(self, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle linear_op task: compress matrix, apply to input, use adapter cluster."""
        matrix = task.get("matrix")
        inp = task.get("input")

        if matrix is None or inp is None:
            return {"id": task_id, "status": "error", "message": "Missing matrix or input"}

        compressed = self.flop_compressor.compress_linear_op(matrix)
        result = self.flop_compressor.apply_compressed_op(compressed, inp)

        task_embedding = self._compute_task_embedding(task)
        self.memory_router.add_entry(
            key=task_id, embedding=task_embedding, payload={"kind": "linear_op", "result_shape": result.shape}
        )

        work_units = [WorkUnit(job_id=task_id, payload={"op": "linear_op"}, cost_hint=10)]
        self.adapter_cluster.submit_work_units(work_units)
        self.adapter_cluster.step()

        return {
            "id": task_id,
            "status": "success",
            "result": result,
            "compression_rank": compressed["compression_rank"],
        }

    def _handle_branch_and_interfere(self, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle branch_and_interfere: spawn branches, interfere, collapse."""
        base_payload = task.get("base_payload", {})
        variations = task.get("variations", [])
        scoring_fn = task.get("scoring_fn") or (lambda b: sum(v for v in b.payload.values() if isinstance(v, (int, float))))
        temperature = task.get("temperature", 0.1)
        top_k = task.get("top_k", 1)

        base_state = BranchState(id=f"{task_id}_base", amplitude=complex(1.0, 0.0), payload=base_payload)
        branches = self.quantum_approx.spawn_branches(base_state, variations)
        interference = self.quantum_approx.interfere(branches, scoring_fn, temperature)
        collapsed = self.quantum_approx.collapse(interference, scoring_fn, top_k)

        task_embedding = self._compute_task_embedding(task)
        self.memory_router.add_entry(
            key=task_id,
            embedding=task_embedding,
            payload={
                "kind": "branch_and_interfere",
                "num_branches": len(branches),
                "num_collapsed": len(collapsed),
            },
        )

        work_units = [
            WorkUnit(job_id=task_id, payload={"branch_id": b.id}, cost_hint=5) for b in branches
        ]
        self.adapter_cluster.submit_work_units(work_units)
        self.adapter_cluster.step()

        return {
            "id": task_id,
            "status": "success",
            "branches": [{"id": b.id, "amplitude": complex(b.amplitude), "payload": b.payload} for b in collapsed],
        }

    def _handle_cached_function(self, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cached_function: cache expensive computation."""
        cache_key = task.get("cache_key", task_id)
        fn = task.get("fn")

        if fn is None:
            return {"id": task_id, "status": "error", "message": "Missing fn"}

        result = self.flop_compressor.cached_function(cache_key, fn)

        task_embedding = self._compute_task_embedding(task)
        self.memory_router.add_entry(
            key=task_id, embedding=task_embedding, payload={"kind": "cached_function", "cache_key": cache_key}
        )

        return {"id": task_id, "status": "success", "result": result}

    def _compute_task_embedding(self, task: Dict[str, Any]) -> np.ndarray:
        """Generate deterministic embedding for a task."""
        kind = task.get("kind", "")
        task_id = task.get("id", "")
        content = f"{kind}:{task_id}"
        content_hash = hashlib.sha256(content.encode()).digest()
        raw_embedding = np.frombuffer(content_hash[:self.dim * 4], dtype=np.float32)

        if len(raw_embedding) < self.dim:
            raw_embedding = np.pad(raw_embedding, (0, self.dim - len(raw_embedding)), constant_values=0.0)
        else:
            raw_embedding = raw_embedding[: self.dim]

        norm = np.linalg.norm(raw_embedding) or 1.0
        return raw_embedding / norm


__all__ = ["SyntheticQuantumGPU"]
