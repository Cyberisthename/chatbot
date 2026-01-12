"""src/multiversal/multiversal_protein_computer.py

Multiversal computing for protein folding: run parallel "universes" each with
different random initialization, optimization strategy, or parameter choices.

Each "universe" corresponds to a separate folding trajectory in conformational
space. The system runs them in parallel (via multiprocessing for true CPU
concurrency, avoiding the Python GIL).

This is a real parallel computing approach, not mock/fake.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protein_folding_engine import (
    FoldingParameters,
    ProteinFoldingEngine,
    ProteinStructure,
)


logger = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    """Configuration for a single universe's protein folding run."""

    universe_id: str
    seed: int
    steps: int = 5000
    t_start: float = 2.0
    t_end: float = 0.1
    params_override: Optional[Dict[str, Any]] = None


@dataclass
class UniverseResult:
    """Result from a single universe."""

    universe_id: str
    seed: int
    best_energy: float
    final_energy: float
    acceptance_rate: float
    runtime_s: float
    best_structure: ProteinStructure
    trajectory: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiversalResult:
    """Aggregate result from all universes."""

    sequence: str
    n_universes: int
    universes: List[UniverseResult]
    best_overall: UniverseResult
    energy_mean: float
    energy_std: float
    total_runtime_s: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "n_universes": self.n_universes,
            "best_overall_universe_id": self.best_overall.universe_id,
            "best_overall_energy": self.best_overall.best_energy,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "total_runtime_s": self.total_runtime_s,
            "timestamp": self.timestamp,
            "universes": [
                {
                    "universe_id": u.universe_id,
                    "seed": u.seed,
                    "best_energy": u.best_energy,
                    "final_energy": u.final_energy,
                    "acceptance_rate": u.acceptance_rate,
                    "runtime_s": u.runtime_s,
                    "metadata": u.metadata,
                }
                for u in self.universes
            ],
        }


def _fold_universe_worker(
    sequence: str,
    config: UniverseConfig,
    artifacts_dir: Path,
) -> UniverseResult:
    """Top-level worker function for multiprocessing."""
    logger.info("[%s] Starting fold (seed=%d, steps=%d)", config.universe_id, config.seed, config.steps)
    start = time.time()

    # Build engine
    params = FoldingParameters()
    if config.params_override:
        for k, v in config.params_override.items():
            setattr(params, k, v)

    engine = ProteinFoldingEngine(artifacts_dir=artifacts_dir, params=params)

    initial = engine.initialize_extended_chain(sequence, seed=config.seed)

    result = engine.metropolis_anneal(
        initial,
        steps=config.steps,
        t_start=config.t_start,
        t_end=config.t_end,
        seed=config.seed,
    )

    runtime = time.time() - start

    universe_result = UniverseResult(
        universe_id=config.universe_id,
        seed=config.seed,
        best_energy=result["best_energy"],
        final_energy=result["final_energy"],
        acceptance_rate=result["acceptance_rate"],
        runtime_s=runtime,
        best_structure=result["best_structure"],
        trajectory=result["trajectory"],
        metadata={
            "steps": config.steps,
            "t_start": config.t_start,
            "t_end": config.t_end,
        },
    )
    logger.info("[%s] Fold complete: E=%.6f (runtime=%.3fs)", config.universe_id, universe_result.best_energy, runtime)
    return universe_result


class MultiversalProteinComputer:
    """Compute protein folding across parallel universes."""

    def __init__(
        self,
        artifacts_dir: str | Path = "./protein_folding_artifacts",
        log_level: int = logging.INFO,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging(log_level)

    def _setup_logging(self, log_level: int):
        # Only setup if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def fold_multiversal(
        self,
        sequence: str,
        n_universes: int = 4,
        steps_per_universe: int = 5000,
        t_start: float = 2.0,
        t_end: float = 0.1,
        base_seed: int = 42,
        max_workers: Optional[int] = None,
        save_artifacts: bool = True,
    ) -> MultiversalResult:
        """
        Fold a sequence across multiple parallel universes using ProcessPoolExecutor.
        """
        logger.info("==== MULTIVERSAL PROTEIN FOLDING START (ProcessPool) ====")
        logger.info("Sequence: %s (length=%d)", sequence, len(sequence))
        logger.info("Universes: %d | Steps/universe: %d", n_universes, steps_per_universe)

        start_time = time.time()

        # Build universe configs
        configs = []
        for i in range(n_universes):
            cfg = UniverseConfig(
                universe_id=f"universe_{i:03d}",
                seed=base_seed + i,
                steps=steps_per_universe,
                t_start=t_start,
                t_end=t_end,
            )
            configs.append(cfg)

        # Run universes in parallel using processes to bypass GIL
        results: List[UniverseResult] = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_fold_universe_worker, sequence, cfg, self.artifacts_dir): cfg 
                for cfg in configs
            }

            for future in concurrent.futures.as_completed(future_map):
                cfg = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        "Universe %s complete: best_E=%.6f runtime=%.3fs acc=%.3f",
                        result.universe_id,
                        result.best_energy,
                        result.runtime_s,
                        result.acceptance_rate,
                    )
                except Exception as exc:
                    logger.error("Universe %s failed: %s", cfg.universe_id, exc)

        total_time = time.time() - start_time
        if not results:
            raise RuntimeError("All universe computations failed.")

        # Aggregate
        best = min(results, key=lambda r: r.best_energy)
        energies = [r.best_energy for r in results]
        mean_e = sum(energies) / len(energies)
        std_e = (sum((e - mean_e) ** 2 for e in energies) / len(energies)) ** 0.5 if len(energies) > 1 else 0.0

        multiversal_result = MultiversalResult(
            sequence=sequence,
            n_universes=n_universes,
            universes=results,
            best_overall=best,
            energy_mean=mean_e,
            energy_std=std_e,
            total_runtime_s=total_time,
            timestamp=time.time(),
        )

        logger.info("==== MULTIVERSAL PROTEIN FOLDING COMPLETE ====")
        logger.info("Best universe: %s | E=%.6f", best.universe_id, best.best_energy)
        logger.info("Total runtime: %.3f s", total_time)

        if save_artifacts:
            self._save_multiversal_artifact(multiversal_result)

        return multiversal_result

    def _save_multiversal_artifact(self, result: MultiversalResult) -> str:
        """Save the multiversal result (all universes) as a JSON artifact."""
        timestamp = int(time.time())
        filename = f"multiversal_fold_{timestamp}.json"
        path = self.artifacts_dir / filename

        payload = result.to_dict()

        # Add best structure
        best_st = result.best_overall.best_structure
        payload["best_structure"] = best_st.to_dict()

        # Save
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Multiversal artifact saved: %s", path)
        return str(path)
