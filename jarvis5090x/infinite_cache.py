from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional


class InfiniteMemoryCache:
    """
    A never-recompute cache backed by deterministic hashing.
    Provides instant lookup of previously computed results.
    """

    def __init__(self, max_items: int = 200000) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def lookup(self, op_signature: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._compute_key(op_signature, payload)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return copy.deepcopy(result)
        self._misses += 1
        return None

    def store(self, op_signature: str, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        key = self._compute_key(op_signature, payload)
        self._cache[key] = copy.deepcopy(result)
        self._cache.move_to_end(key)
        
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "cached_items": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def _compute_key(self, op_signature: str, payload: Dict[str, Any]) -> str:
        cleaned_payload = {k: v for k, v in payload.items() if not k.startswith("__")}
        canonical = json.dumps(
            self._normalize(cleaned_payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        combined = f"{op_signature}:{canonical}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [self._normalize(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize(v) for v in value]
        if isinstance(value, set):
            return [self._normalize(v) for v in sorted(value, key=lambda item: str(item))]
        if isinstance(value, bytes):
            return {"__bytes__": value.hex()}
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (str, bool)) or value is None:
            return value
        return str(value)
