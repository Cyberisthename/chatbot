"""FLOP Compression Layer - Cache and compress expensive operations."""

import functools
import hashlib
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import numpy as np

from .config import DEFAULT_MAX_CACHE_ITEMS


class FlopCompressor:
    """Compresses and caches expensive linear operations and function results."""

    def __init__(self, max_cache_items: int = DEFAULT_MAX_CACHE_ITEMS) -> None:
        """Initialize the FLOP compressor.

        Args:
            max_cache_items: Maximum number of cached items (LRU eviction).
        """
        self.max_cache_items = max_cache_items
        self._lru_cache: OrderedDict[str, Any] = OrderedDict()
        self._compression_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _hash_array(self, arr: np.ndarray) -> str:
        """Generate a deterministic hash for a numpy array."""
        arr_bytes = arr.tobytes()
        return hashlib.sha256(arr_bytes).hexdigest()[:16]

    def _evict_if_needed(self, cache: OrderedDict) -> None:
        """Evict oldest entry if cache exceeds max size."""
        if len(cache) >= self.max_cache_items:
            cache.popitem(last=False)

    def compress_linear_op(self, matrix: np.ndarray, rank: Optional[int] = None) -> Dict[str, Any]:
        """Compress a linear operator using low-rank SVD approximation.

        Args:
            matrix: Dense matrix to compress (shape M x N).
            rank: Target rank for compression. If None, auto-select based on singular values.

        Returns:
            Dictionary containing compressed representation with keys:
                - 'u': Left singular vectors (M x rank)
                - 's': Singular values (rank,)
                - 'vt': Right singular vectors (rank x N)
                - 'original_shape': Original matrix shape
                - 'compression_rank': Effective rank used
        """
        matrix_hash = self._hash_array(matrix)

        if matrix_hash in self._compression_cache:
            self._compression_cache.move_to_end(matrix_hash)
            return self._compression_cache[matrix_hash]

        u, s, vt = np.linalg.svd(matrix, full_matrices=False)

        if rank is None:
            threshold = 0.01 * s[0]
            rank = np.sum(s > threshold)
            rank = max(1, min(rank, len(s)))

        compressed = {
            "u": u[:, :rank],
            "s": s[:rank],
            "vt": vt[:rank, :],
            "original_shape": matrix.shape,
            "compression_rank": rank,
            "hash": matrix_hash,
        }

        self._evict_if_needed(self._compression_cache)
        self._compression_cache[matrix_hash] = compressed

        return compressed

    def apply_compressed_op(self, compressed: Dict[str, Any], x: np.ndarray) -> np.ndarray:
        """Apply a compressed linear operator to a vector or batch of vectors.

        Args:
            compressed: Compressed representation from compress_linear_op.
            x: Input vector/matrix of shape (N,) or (N, batch_size).

        Returns:
            Result of applying the compressed operator: (M,) or (M, batch_size).
        """
        u = compressed["u"]
        s = compressed["s"]
        vt = compressed["vt"]

        if x.ndim == 1:
            result = u @ (s * (vt @ x))
        else:
            result = u @ (s[:, None] * (vt @ x))

        return result

    def cached_function(self, key: str, fn: Callable[[], Any]) -> Any:
        """Cache the result of an expensive function call.

        Args:
            key: Unique cache key.
            fn: Function to call if not cached (no arguments).

        Returns:
            Cached or computed result.
        """
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
            return self._lru_cache[key]

        result = fn()

        self._evict_if_needed(self._lru_cache)
        self._lru_cache[key] = result

        return result

    def clear(self) -> None:
        """Clear all caches."""
        self._lru_cache.clear()
        self._compression_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "lru_cache_size": len(self._lru_cache),
            "compression_cache_size": len(self._compression_cache),
            "max_cache_items": self.max_cache_items,
        }


_global_compressor: Optional[FlopCompressor] = None


def get_global_compressor() -> FlopCompressor:
    """Get or create the global FlopCompressor instance."""
    global _global_compressor
    if _global_compressor is None:
        _global_compressor = FlopCompressor()
    return _global_compressor


def flop_cached(key: str) -> Callable:
    """Decorator to cache expensive function results using the global FlopCompressor.

    Args:
        key: Cache key for this function call.

    Example:
        @flop_cached("expensive_computation_v1")
        def expensive_fn():
            return compute_something_slow()
    """

    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        @functools.wraps(fn)
        def wrapper() -> Any:
            compressor = get_global_compressor()
            return compressor.cached_function(key, fn)

        return wrapper

    return decorator


__all__ = ["FlopCompressor", "flop_cached", "get_global_compressor"]
