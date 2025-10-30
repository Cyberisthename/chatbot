"""
Quantum Experiments Collection

A collection of standalone quantum physics experiments demonstrating
key concepts in quantum mechanics.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["quick_interference"]

EXPERIMENTS_DIR = Path(__file__).parent
ARTIFACTS_DIR = EXPERIMENTS_DIR.parent / "artifacts"
