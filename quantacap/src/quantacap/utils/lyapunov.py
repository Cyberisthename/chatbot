"""Utilities for enforcing Lyapunov-style monotonicity guards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LyapunovGuard:
    """Simple Lyapunov guard enforcing non-increasing energy-like metric.

    The guard keeps track of a scalar potential :math:`V`. Updates are accepted
    only when ``new_V <= previous_V + tolerance``.  This lightweight helper is
    sufficient for the synthetic experiments where we monitor convergence and
    forbid destabilising jumps before committing state to the adapter store.
    """

    tolerance: float = 1e-9
    initial: float = float("inf")

    def __post_init__(self) -> None:
        self._current = self.initial

    def allows(self, value: float) -> bool:
        """Return ``True`` if the candidate value keeps the sequence stable."""

        if value <= self._current + self.tolerance:
            self._current = value
            return True
        return False

    def assert_allows(self, value: float, context: str | None = None) -> None:
        """Raise ``RuntimeError`` when the guard would reject ``value``."""

        if not self.allows(value):
            ctx = f" ({context})" if context else ""
            raise RuntimeError(f"Lyapunov guard violation{ctx}: ΔV > 0")

    @property
    def current(self) -> float:
        return self._current
