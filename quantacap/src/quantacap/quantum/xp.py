"""Backend selection helpers for Quantacap quantum modules."""
from __future__ import annotations

from typing import Any

import numpy as _np


def get_xp(use_gpu: bool = False) -> Any:
    """Return the array module for the requested backend.

    Parameters
    ----------
    use_gpu:
        If ``True`` an attempt is made to import CuPy. If the import fails the
        function silently falls back to NumPy so CPU execution remains
        functional on systems without GPU support.
    """

    if use_gpu:
        try:  # pragma: no cover - GPU optional
            import cupy as cp  # type: ignore

            return cp
        except Exception:  # pragma: no cover - import guard
            pass
    return _np


def to_numpy(xp: Any, array: Any):
    """Convert an array from ``xp`` to a NumPy ``ndarray``."""

    if xp is _np:
        return _np.asarray(array)
    # CuPy exposes ``asnumpy`` for host transfer. For other libraries the
    # attribute check avoids hard dependency.
    as_numpy = getattr(xp, "asnumpy", None)
    if callable(as_numpy):  # pragma: no branch - attribute lookup
        return as_numpy(array)
    return _np.asarray(array)
