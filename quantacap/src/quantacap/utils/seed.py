"""Deterministic seed management for reproducibility."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def set_seed(seed: int, np_module: np | None = None) -> None:
    """Set random seeds for both Python random and NumPy.
    
    Args:
        seed: Integer seed for reproducibility
        np_module: NumPy module reference (if available)
    """
    random.seed(seed)
    if np_module is not None:
        np_module.random.seed(seed)
