"""Backend selection utilities for NumPy / CuPy interoperability."""
from __future__ import annotations

import numpy as _np

try:  # pragma: no cover - optional dependency
    import cupy as _cp  # type: ignore
except Exception:  # pragma: no cover - CuPy not available
    _cp = None


def get_array_module(obj=None):
    """Return the array module (NumPy or CuPy) that matches *obj*.

    Parameters
    ----------
    obj:
        Optional array-like instance used to detect whether we are working with
        a GPU-backed CuPy array.
    """

    if obj is not None and _cp is not None and isinstance(obj, _cp.ndarray):
        return _cp
    return _np


def asnumpy(arr):
    """Convert *arr* to a NumPy ``ndarray`` regardless of backend."""

    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return _np.asarray(arr)


__all__ = ["get_array_module", "asnumpy"]
