from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class _VectorizedPayload:
    key_order: Tuple[str, ...]
    segments: Tuple[int, ...]
    values: List[float]


@dataclass(slots=True)
class _BasisEntry:
    key_order: Tuple[str, ...]
    segments: Tuple[int, ...]
    mean_vector: List[float]
    samples: List[List[float]] = field(default_factory=list)
    stable: bool = False
    stability_threshold: int = 4
    tolerance: float = 1e-8

    def add_sample(self, values: List[float]) -> None:
        self.samples.append(values)
        if len(self.samples) > max(self.stability_threshold, 1):
            self.samples.pop(0)
        count = len(self.samples)
        if not self.samples:
            return
        dims = len(self.samples[0])
        self.mean_vector = [
            sum(sample[i] for sample in self.samples) / count
            for i in range(dims)
        ]
        if count >= self.stability_threshold:
            max_dev = 0.0
            for sample in self.samples:
                for idx in range(dims):
                    dev = abs(sample[idx] - self.mean_vector[idx])
                    if dev > max_dev:
                        max_dev = dev
            self.stable = max_dev <= self.tolerance

    def project(self, values: List[float]) -> Optional[float]:
        if len(values) != len(self.mean_vector):
            return None
        norm_sq = sum(component * component for component in self.mean_vector)
        if norm_sq <= self.tolerance:
            return 0.0
        dot = sum(values[i] * self.mean_vector[i] for i in range(len(values)))
        coeff = dot / norm_sq
        residual = [values[i] - coeff * self.mean_vector[i] for i in range(len(values))]
        residual_max = max(abs(val) for val in residual) if residual else 0.0
        if residual_max > self.tolerance * 10:
            return None
        return coeff


class FlopCompressionLayer:
    """Deterministic low-rank approximation helper for repeated operations."""

    def __init__(
        self,
        max_bases: int = 5000,
        stability_threshold: int = 4,
        tolerance: float = 1e-8,
    ) -> None:
        self.max_bases = max_bases
        self.stability_threshold = max(2, stability_threshold)
        self.tolerance = tolerance
        self._bases: Dict[str, _BasisEntry] = {}
        self._observations: Dict[str, int] = {}

    def maybe_compress(self, op_type: str, signature: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        vectorized = self._vectorize(payload)
        if vectorized is None:
            return payload

        basis_key = self._basis_key(op_type, signature)
        basis = self._bases.get(basis_key)
        values = vectorized.values

        if basis is None:
            if len(self._bases) >= self.max_bases:
                return payload
            basis = _BasisEntry(
                key_order=vectorized.key_order,
                segments=vectorized.segments,
                mean_vector=list(values),
                stability_threshold=self.stability_threshold,
                tolerance=self.tolerance,
            )
            basis.add_sample(list(values))
            self._bases[basis_key] = basis
            self._observations[basis_key] = 1
            return payload

        if basis.key_order != vectorized.key_order or basis.segments != vectorized.segments:
            return payload

        basis.add_sample(list(values))
        self._observations[basis_key] = self._observations.get(basis_key, 0) + 1
        if not basis.stable:
            return payload

        coefficient = basis.project(list(values))
        if coefficient is None:
            return payload

        compression_info = {
            "basis_key": basis_key,
            "coefficients": [round(coefficient, 12)],
            "key_order": list(basis.key_order),
            "segments": list(basis.segments),
            "sample_count": len(basis.samples),
        }
        compressed_payload = {**payload}
        compressed_payload["__compressed__"] = True
        compressed_payload["__compression__"] = compression_info
        return compressed_payload

    def has_basis(self, op_type: str, signature: str) -> bool:
        return self._basis_key(op_type, signature) in self._bases

    def stats(self) -> Dict[str, Any]:
        return {
            "basis_count": len(self._bases),
            "observations": dict(self._observations),
            "stable_bases": sum(1 for basis in self._bases.values() if basis.stable),
        }

    def _basis_key(self, op_type: str, signature: str) -> str:
        data = f"{op_type}:{signature}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:24]

    def _vectorize(self, payload: Dict[str, Any]) -> Optional[_VectorizedPayload]:
        numeric_values: List[float] = []
        key_order: List[str] = []
        segments: List[int] = []
        for key in sorted(payload.keys()):
            if key.startswith("__"):
                continue
            value = payload[key]
            flattened = self._flatten_numeric(value)
            if not flattened:
                continue
            key_order.append(str(key))
            segments.append(len(flattened))
            numeric_values.extend(flattened)
        if not numeric_values:
            return None
        return _VectorizedPayload(
            key_order=tuple(key_order),
            segments=tuple(segments),
            values=numeric_values,
        )

    def _flatten_numeric(self, value: Any) -> List[float]:
        if isinstance(value, bool):
            return [1.0 if value else 0.0]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            result: List[float] = []
            for entry in value:
                result.extend(self._flatten_numeric(entry))
            return result
        return []
