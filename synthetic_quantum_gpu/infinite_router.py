"""Infinite Memory Router - deterministic routing over memory entries."""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .config import DEFAULT_MAX_MEMORY_ENTRIES
from .types import MemoryEntry


class InfiniteMemoryRouter:
    """Routes queries to memory entries using deterministic similarity search."""

    def __init__(self, dim: int, max_entries: int = DEFAULT_MAX_MEMORY_ENTRIES) -> None:
        self.dim = dim
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        self._next_index = 0

    def add_entry(self, key: str, embedding: np.ndarray, payload: Dict[str, object]) -> None:
        """Add a memory entry with given embedding and payload."""
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != router dim {self.dim}")
        entry = MemoryEntry(
            key=key,
            embedding=embedding.astype(float),
            payload=payload,
            created_at=time.time(),
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        self._next_index += 1

    def route(self, query_embedding: np.ndarray, top_k: int = 8) -> List[MemoryEntry]:
        """Route query to top-k memory entries using cosine similarity."""
        if not self._entries:
            return []
        if query_embedding.shape[0] != self.dim:
            raise ValueError(f"Query dimension {query_embedding.shape[0]} != router dim {self.dim}")

        query = query_embedding.astype(float)
        q_norm = np.linalg.norm(query) or 1.0
        query /= q_norm

        similarities = []
        for idx, entry in enumerate(self._entries):
            emb = entry.embedding
            e_norm = np.linalg.norm(emb) or 1.0
            sim = float(np.dot(query, emb / e_norm))
            similarities.append((sim, idx, entry))

        similarities.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in similarities[:top_k]]
        return selected

    def snapshot(self) -> Dict[str, object]:
        """Create a deterministic snapshot of router state."""
        return {
            "dim": self.dim,
            "max_entries": self.max_entries,
            "entries": [
                {
                    "key": entry.key,
                    "embedding": entry.embedding.tolist(),
                    "payload": entry.payload,
                    "created_at": entry.created_at,
                }
                for entry in self._entries
            ],
        }

    def load_snapshot(self, snapshot: Dict[str, object]) -> None:
        """Restore router state from snapshot."""
        self.dim = int(snapshot["dim"])
        self.max_entries = int(snapshot["max_entries"])
        self._entries = [
            MemoryEntry(
                key=item["key"],
                embedding=np.array(item["embedding"], dtype=float),
                payload=item["payload"],
                created_at=item["created_at"],
            )
            for item in snapshot["entries"]
        ]


__all__ = ["InfiniteMemoryRouter"]
