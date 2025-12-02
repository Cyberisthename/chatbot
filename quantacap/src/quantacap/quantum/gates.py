"""Gate factories supporting CPU/GPU backends."""
from __future__ import annotations

import math
from typing import Any

from .xp import get_xp


def _resolve_backend(xp: Any | None):
    return xp if xp is not None else get_xp(False)


def _resolve_dtype(xp: Any, dtype: str):
    return xp.complex128 if dtype == "complex128" else xp.complex64


def kron_n(*mats, xp: Any | None = None):
    xp = _resolve_backend(xp)
    if not mats:
        raise ValueError("kron_n requires at least one matrix")
    out = xp.array([[1.0]], dtype=xp.complex128)
    for mat in mats:
        out = xp.kron(out, xp.asarray(mat, dtype=xp.complex128))
    return out


def I(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.eye(2, dtype=dt)


def X(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dt)


def Y(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.array([[0.0, -1j], [1j, 0.0]], dtype=dt)


def Z(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dt)


def H(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    factor = 1.0 / math.sqrt(2.0)
    return factor * xp.array([[1.0, 1.0], [1.0, -1.0]], dtype=dt)


def RX(theta: float, *, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    half = theta / 2.0
    return xp.array(
        [
            [math.cos(half), -1j * math.sin(half)],
            [-1j * math.sin(half), math.cos(half)],
        ],
        dtype=dt,
    )


def RY(theta: float, *, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    half = theta / 2.0
    return xp.array(
        [
            [math.cos(half), -math.sin(half)],
            [math.sin(half), math.cos(half)],
        ],
        dtype=dt,
    )


def RZ(theta: float, *, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    phase = theta / 2.0
    e_neg = math.cos(phase) - 1j * math.sin(phase)
    e_pos = math.cos(phase) + 1j * math.sin(phase)
    return xp.array(
        [
            [e_neg, 0.0],
            [0.0, e_pos],
        ],
        dtype=dt,
    )


def CNOT(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dt,
    )


def SWAP(*, xp: Any | None = None, dtype: str = "complex128"):
    xp = _resolve_backend(xp)
    dt = _resolve_dtype(xp, dtype)
    return xp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dt,
    )
