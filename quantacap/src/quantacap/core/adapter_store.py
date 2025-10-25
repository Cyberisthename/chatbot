import json, os, time
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
    if "n" not in state or "amps" not in state:
        raise ValueError("state payload requires 'n' and 'amps'")
    amps = np.asarray(state["amps"], dtype=np.complex128).reshape(-1)
    probs = state.get("probs")
    if probs is None:
        probs = (np.abs(amps) ** 2).tolist()
    else:
        probs = list(map(float, probs))
    return {
        "n": int(state["n"]),
        "amps": _serialize_amplitudes(amps),
        "probs": probs,
    }


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


