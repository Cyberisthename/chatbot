"""Quantum Approximation Layer - emulate quantum branching and interference."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Dict, Iterable, List

import numpy as np

from .config import DEFAULT_SEED
from .types import BranchState, InterferenceResult


class QuantumApproximator:
    """Approximates quantum-style branching and interference deterministically."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        """Initialize approximator with deterministic random generator."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def spawn_branches(
        self,
        base_state: BranchState,
        variations: List[Dict[str, object]],
    ) -> List[BranchState]:
        """Spawn new branches from a base state with given variations.

        The payload of the base state is copied and updated with each variation.
        Amplitudes are perturbed deterministically with small complex noise while
        ensuring the total power of new branches equals the base branch power.
        """
        if not variations:
            return [base_state]

        num = len(variations)
        base_amp = base_state.amplitude
        base_payload = base_state.payload

        # Normalize amplitude magnitude across branches
        magnitude = np.abs(base_amp)
        target_mag = magnitude / math.sqrt(num)
        phase = np.angle(base_amp)

        branches: List[BranchState] = []
        for idx, variation in enumerate(variations):
            # Deterministic perturbation based on seed, index, and branch id
            perturb_real = self._rng.normal(loc=1.0, scale=0.01)
            perturb_imag = self._rng.normal(loc=0.0, scale=0.01)
            perturb = complex(perturb_real, perturb_imag)
            amplitude = target_mag * perturb * np.exp(1j * phase)

            payload = {**base_payload}
            payload.update(variation)

            branch = BranchState(
                id=f"{base_state.id}_b{idx}",
                amplitude=amplitude,
                payload=payload,
            )
            branches.append(branch)

        return branches

    def interfere(
        self,
        branches: List[BranchState],
        scoring_fn: Callable[[BranchState], float],
        temperature: float = 0.1,
    ) -> InterferenceResult:
        """Interfere branches using scoring function and softmax weighting.

        Args:
            branches: List of branch states to interfere.
            scoring_fn: Callable returning real-valued score for each branch.
            temperature: Softmax temperature controlling sharpness.
        """
        if not branches:
            return InterferenceResult(branches=[], normalized=True)

        temp = max(temperature, 1e-6)
        scores = np.array([scoring_fn(branch) for branch in branches], dtype=float)
        amplitudes = np.array([branch.amplitude for branch in branches], dtype=complex)
        magnitudes_sq = np.abs(amplitudes) ** 2

        logits = (scores * magnitudes_sq) / temp
        logits -= logits.max()
        weights = np.exp(logits)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        probabilities = weights / weights.sum()

        normalized_branches: List[BranchState] = []
        for branch, prob in zip(branches, probabilities):
            phase = np.angle(branch.amplitude)
            new_amp = math.sqrt(prob) * np.exp(1j * phase)
            normalized_branches.append(replace(branch, amplitude=new_amp))

        return InterferenceResult(branches=normalized_branches, normalized=True)

    def collapse(
        self,
        interference: InterferenceResult,
        scoring_fn: Callable[[BranchState], float],
        top_k: int = 1,
    ) -> List[BranchState]:
        """Collapse interference result to top-k branches deterministically."""
        branches = interference.branches
        if not branches:
            return []

        scored = [
            (np.abs(branch.amplitude) ** 2 * scoring_fn(branch), idx, branch)
            for idx, branch in enumerate(branches)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected = [item[2] for item in scored[:top_k]]
        total_power = sum(np.abs(branch.amplitude) ** 2 for branch in selected) or 1.0
        normalized = [
            replace(branch, amplitude=branch.amplitude / math.sqrt(total_power))
            for branch in selected
        ]
        return normalized

    def reseed(self, seed: int) -> None:
        """Reseed the internal random generator."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)


__all__ = ["QuantumApproximator"]
