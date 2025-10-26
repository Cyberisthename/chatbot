"""Persistence helpers for experiment adapters."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

import numpy as np

ROOT = ".adapters"


def _serialize_amplitudes(amps: Iterable[complex]) -> list[list[float]]:
    serialized: list[list[float]] = []
    for amp in amps:
        comp = complex(amp)
        serialized.append([float(np.real(comp)), float(np.imag(comp))])
    return serialized


def _prepare_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    if "n" not in state:
        raise ValueError("state payload requires 'n'")
    n = int(state["n"])
    dtype = str(state.get("dtype", "complex128"))

    amps_serialized: list[list[float]] | None = None
    probs_list: list[float] | None = None

    if "amps" in state and state["amps"] is not None:
        amps = np.asarray(state["amps"], dtype=np.complex128).reshape(-1)
        amps_serialized = _serialize_amplitudes(amps)
        probs_list = (np.abs(amps) ** 2).real.tolist()

    if "probs" in state and state["probs"] is not None:
        probs_list = list(map(float, state["probs"]))

    if probs_list is None:
        raise ValueError("state payload requires 'probs' or 'amps'")

    payload: dict[str, Any] = {
        "n": n,
        "dtype": dtype,
        "probs": probs_list,
    }
    if amps_serialized is not None:
        payload["amps"] = amps_serialized

    backend = state.get("backend")
    if backend is not None:
        payload["backend"] = backend

    return payload


def create_adapter(id, data=None, *, state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f"{id}.json")
    payload: dict[str, Any] = {"id": id, "ts": time.time()}
    if data is not None:
        payload["data"] = data
    if state is not None:
        payload["state"] = _prepare_state_payload(state)
    if meta is not None:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_adapter(id):
    path = os.path.join(ROOT, f"{id}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_adapters(prefix: str | None = None) -> list[str]:
    if not os.path.isdir(ROOT):
        return []
    items: list[str] = []
    for name in os.listdir(ROOT):
        if not name.endswith(".json"):
            continue
        if prefix and not name.startswith(prefix):
            continue
        items.append(name[:-5])
    items.sort()
    return items


