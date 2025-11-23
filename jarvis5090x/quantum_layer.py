from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Branch:
    amplitude: complex
    state: Dict[str, Any]

    def probability(self) -> float:
        return abs(self.amplitude) ** 2


@dataclass
class QuantumExperimentHooks:
    on_spawn: Optional[Callable[[List[Branch], Dict[str, Any]], None]] = None
    on_interfere: Optional[Callable[[List[Branch], Dict[str, Any]], None]] = None
    on_collapse: Optional[Callable[[List[Branch], Dict[str, Any]], None]] = None


class QuantumApproximationLayer:
    """Quantum-inspired approximation layer with deterministic behaviour."""

    def __init__(
        self,
        max_branches: int = 64,
        seed: int = 42,
        event_hooks: Optional[QuantumExperimentHooks] = None,
    ) -> None:
        self.max_branches = max_branches
        self.seed = seed
        self._hooks = event_hooks

    def spawn(
        self,
        base_state: Dict[str, Any],
        variations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        hooks: Optional[QuantumExperimentHooks] = None,
    ) -> List[Branch]:
        variations = variations[: max(0, self.max_branches - 1)]
        total = len(variations) + 1
        if total == 0:
            return []

        amplitude_scale = 1.0 / math.sqrt(total)
        branches: List[Branch] = [
            Branch(amplitude=complex(amplitude_scale, 0.0), state=copy.deepcopy(base_state))
        ]

        for index, variation in enumerate(variations):
            merged_state = copy.deepcopy(base_state)
            merged_state.update(variation)
            phase = self._phase_for_state(merged_state, index)
            amplitude = amplitude_scale * complex(math.cos(phase), math.sin(phase))
            branches.append(Branch(amplitude=amplitude, state=merged_state))

        branches = self._normalize_branches(branches)
        self._emit(hooks, "on_spawn", branches, metadata or {})
        return branches

    def interfere(
        self,
        branches: List[Branch],
        scoring_fn: Callable[[Dict[str, Any]], float],
        metadata: Optional[Dict[str, Any]] = None,
        hooks: Optional[QuantumExperimentHooks] = None,
    ) -> List[Branch]:
        if not branches:
            return []

        combined: Dict[str, complex] = {}
        representative: Dict[str, Dict[str, Any]] = {}

        for branch in branches:
            score = max(float(scoring_fn(branch.state)), 0.0)
            adjusted_amplitude = branch.amplitude * (score + 1e-12)
            signature = self._state_signature(branch.state)
            combined[signature] = combined.get(signature, 0j) + adjusted_amplitude
            representative[signature] = branch.state

        interfered = [
            Branch(amplitude=combined[sig], state=representative[sig])
            for sig in sorted(combined.keys())
        ]
        result = self._normalize_branches(interfered)
        self._emit(hooks, "on_interfere", result, metadata or {})
        return result

    def collapse(
        self,
        branches: List[Branch],
        top_k: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        hooks: Optional[QuantumExperimentHooks] = None,
    ) -> Dict[str, Any]:
        if not branches:
            return {}

        normalized = self._normalize_branches(branches)
        ordered = sorted(normalized, key=lambda b: b.probability(), reverse=True)
        selected = ordered[: max(1, top_k)]

        if top_k <= 1:
            collapsed = copy.deepcopy(selected[0].state)
        else:
            weights = [branch.probability() for branch in selected]
            weight_sum = sum(weights)
            if weight_sum <= 0:
                collapsed = copy.deepcopy(selected[0].state)
            else:
                normalized_weights = [w / weight_sum for w in weights]
                collapsed = self._blend_states(
                    (branch.state for branch in selected),
                    normalized_weights,
                )

        collapse_metadata = dict(metadata or {})
        collapse_metadata.setdefault("top_k", top_k)
        collapse_metadata["collapsed_state"] = copy.deepcopy(collapsed)
        self._emit(hooks, "on_collapse", selected, collapse_metadata)
        return collapsed

    def _blend_states(
        self,
        states: Iterable[Dict[str, Any]],
        weights: Iterable[float],
    ) -> Dict[str, Any]:
        aggregated: Dict[str, float] = {}
        totals: Dict[str, float] = {}
        chosen: Optional[Dict[str, Any]] = None
        weight_list = list(weights)
        state_list = list(states)
        if not state_list:
            return {}
        if chosen is None and state_list:
            chosen = copy.deepcopy(state_list[0])

        for state, weight in zip(state_list, weight_list):
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight
                    totals[key] = totals.get(key, 0.0) + weight
        result = chosen if chosen is not None else {}
        for key, total in totals.items():
            if total > 0:
                result[key] = aggregated[key] / total
        return result

    def _phase_for_state(self, state: Dict[str, Any], index: int) -> float:
        payload = json.dumps(self._normalize_value(state), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{self.seed}:{index}:{payload}".encode("utf-8")).hexdigest()
        bucket = int(digest[:16], 16)
        return (bucket / float(0xFFFFFFFFFFFFFFFF)) * 2.0 * math.pi - math.pi

    def _state_signature(self, state: Dict[str, Any]) -> str:
        payload = json.dumps(self._normalize_value(state), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _normalize_branches(self, branches: List[Branch]) -> List[Branch]:
        total = sum(branch.probability() for branch in branches)
        if total <= 0:
            return branches
        scale = 1.0 / math.sqrt(total)
        return [Branch(amplitude=branch.amplitude * scale, state=branch.state) for branch in branches]

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, set):
            return [self._normalize_value(v) for v in sorted(value, key=lambda item: str(item))]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (str, bool)) or value is None:
            return value
        return str(value)

    def _emit(
        self,
        hooks: Optional[QuantumExperimentHooks],
        event_name: str,
        branches: List[Branch],
        metadata: Dict[str, Any],
    ) -> None:
        hook_set = hooks or self._hooks
        if hook_set is None:
            return
        hook_fn = getattr(hook_set, event_name, None)
        if hook_fn and callable(hook_fn):
            safe_branches = [
                Branch(amplitude=branch.amplitude, state=copy.deepcopy(branch.state))
                for branch in branches
            ]
            hook_fn(safe_branches, copy.deepcopy(metadata))
