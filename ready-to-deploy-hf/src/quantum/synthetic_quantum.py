"""Synthetic Quantum Module for JARVIS-2v
Generates quantum-inspired artifacts and experiments.

This module is designed to run without heavyweight scientific dependencies.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import random
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.adapter_engine import AdapterEngine, QuantumArtifact


@dataclass
class ExperimentConfig:
    """Configuration for synthetic quantum experiments."""

    experiment_type: str
    iterations: int = 1000
    noise_level: float = 0.1
    seed: Optional[int] = None
    parameters: Dict[str, Any] = None


class SyntheticQuantumEngine:
    """Engine for running synthetic quantum experiments and generating artifacts."""

    def __init__(self, artifacts_path: str, adapter_engine: AdapterEngine):
        self.artifacts_path = Path(artifacts_path)
        self.adapter_engine = adapter_engine
        self.experiment_registry = self._load_experiment_registry()
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

    def _load_experiment_registry(self) -> Dict[str, Any]:
        registry_path = self.artifacts_path / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {"experiments": [], "artifacts": []}

    def _save_experiment_registry(self):
        registry_path = self.artifacts_path / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(self.experiment_registry, f, indent=2)

    def run_interference_experiment(self, config: ExperimentConfig) -> QuantumArtifact:
        rng = random.Random(config.seed)

        angles = [
            2.0 * math.pi * i / (config.iterations - 1) if config.iterations > 1 else 0.0
            for i in range(config.iterations)
        ]
        interference_pattern = [
            (math.sin(a) ** 2) + rng.gauss(0.0, config.noise_level) for a in angles
        ]

        mean_intensity = statistics.fmean(interference_pattern) if interference_pattern else 0.0
        min_val = min(interference_pattern) if interference_pattern else 0.0
        max_val = max(interference_pattern) if interference_pattern else 0.0
        visibility = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) != 0 else 0.0
        std_dev = statistics.pstdev(interference_pattern) if len(interference_pattern) > 1 else 0.0

        artifact_id = f"interference_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        results = {
            "pattern": interference_pattern,
            "angles": angles,
            "statistics": {
                "mean_intensity": float(mean_intensity),
                "visibility": float(visibility),
                "std_dev": float(std_dev),
            },
            "histogram": self._generate_histogram(interference_pattern, bins=50),
        }

        linked_adapter = self.adapter_engine.create_adapter(
            task_tags=["quantum", "interference", "physics"],
            y_bits=[0] * 16,
            z_bits=[0] * 8,
            x_bits=[0] * 8,
            parameters={
                "experiment_type": "interference",
                "mean_intensity": mean_intensity,
                "visibility": visibility,
            },
        )

        artifact = QuantumArtifact(
            artifact_id=artifact_id,
            experiment_type="interference_experiment",
            config=config.__dict__,
            results=results,
            linked_adapter_ids=[linked_adapter.id],
        )
        self._save_artifact(artifact)
        return artifact

    def run_bell_pair_simulation(self, config: ExperimentConfig) -> QuantumArtifact:
        rng = random.Random(config.seed)

        measurements: List[Dict[str, int]] = []
        correlations: List[float] = []

        for _ in range(config.iterations):
            alice_meas = 0 if rng.random() < 0.5 else 1
            if rng.random() > config.noise_level:
                bob_meas = alice_meas
            else:
                bob_meas = 1 - alice_meas
            measurements.append({"alice": alice_meas, "bob": bob_meas})
            correlations.append(1.0 if alice_meas == bob_meas else -1.0)

        avg_correlation = statistics.fmean(correlations) if correlations else 0.0

        artifact_id = f"bell_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        results = {
            "measurements": measurements[:100],
            "correlations": correlations,
            "statistics": {
                "average_correlation": float(avg_correlation),
                "bell_inequality_violation": avg_correlation > 0.5,
                "entanglement_fidelity": float(avg_correlation),
            },
            "correlation_matrix": self._generate_correlation_matrix(measurements),
        }

        linked_adapter = self.adapter_engine.create_adapter(
            task_tags=["quantum", "entanglement", "bell"],
            y_bits=[0] * 16,
            z_bits=[0] * 8,
            x_bits=[0] * 8,
            parameters={
                "experiment_type": "bell_pair",
                "correlation": avg_correlation,
                "entanglement": avg_correlation > 0.7,
            },
        )

        artifact = QuantumArtifact(
            artifact_id=artifact_id,
            experiment_type="bell_pair_simulation",
            config=config.__dict__,
            results=results,
            linked_adapter_ids=[linked_adapter.id],
        )
        self._save_artifact(artifact)
        return artifact

    def run_chsh_test(self, config: ExperimentConfig) -> QuantumArtifact:
        rng = random.Random(config.seed)

        angles_a = [0.0, math.pi / 2.0]
        angles_b = [math.pi / 4.0, 3.0 * math.pi / 4.0]

        chsh_values: List[float] = []

        n_blocks = max(1, config.iterations // 4)
        for _ in range(n_blocks):
            results_block: List[float] = []
            for a in angles_a:
                for b in angles_b:
                    expected_correlation = -math.cos(a - b)
                    measured = expected_correlation + rng.gauss(0.0, config.noise_level)
                    results_block.append(measured)

            S = abs(results_block[0] + results_block[1] + results_block[2] - results_block[3])
            chsh_values.append(S)

        avg_S = statistics.fmean(chsh_values) if chsh_values else 0.0
        violation_count = sum(1 for S in chsh_values if S > 2.0)
        violation_ratio = violation_count / len(chsh_values) if chsh_values else 0.0

        artifact_id = f"chsh_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        results = {
            "chsh_values": chsh_values[:100],
            "statistics": {
                "average_S": float(avg_S),
                "violation_ratio": float(violation_ratio),
                "quantum_violation": avg_S > 2.0,
                "max_S": float(max(chsh_values) if chsh_values else 0.0),
                "min_S": float(min(chsh_values) if chsh_values else 0.0),
            },
            "violation_histogram": self._generate_histogram(chsh_values, bins=30),
        }

        linked_adapter = self.adapter_engine.create_adapter(
            task_tags=["quantum", "chsh", "inequality", "nonlocality"],
            y_bits=[0] * 16,
            z_bits=[0] * 8,
            x_bits=[0] * 8,
            parameters={
                "experiment_type": "chsh",
                "violation_ratio": violation_ratio,
                "quantum_behavior": avg_S > 2.0,
            },
        )

        artifact = QuantumArtifact(
            artifact_id=artifact_id,
            experiment_type="chsh_test",
            config=config.__dict__,
            results=results,
            linked_adapter_ids=[linked_adapter.id],
        )
        self._save_artifact(artifact)
        return artifact

    def run_noise_field_scan(self, config: ExperimentConfig) -> QuantumArtifact:
        rng = random.Random(config.seed)

        noise_levels = [i / 49.0 for i in range(50)]
        coherence_measurements: List[float] = []

        for noise in noise_levels:
            coherence = math.exp(-noise * 2.0) + rng.gauss(0.0, 0.1)
            coherence_measurements.append(max(0.0, min(1.0, coherence)))

        coherence_time: float | str = self._estimate_decay_time(noise_levels, coherence_measurements)

        artifact_id = f"noise_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        results = {
            "noise_levels": noise_levels,
            "coherence_measurements": coherence_measurements,
            "statistics": {
                "coherence_time": float(coherence_time) if isinstance(coherence_time, (int, float)) else coherence_time,
                "max_coherence": float(max(coherence_measurements) if coherence_measurements else 0.0),
                "min_coherence": float(min(coherence_measurements) if coherence_measurements else 0.0),
                "avg_coherence": float(statistics.fmean(coherence_measurements) if coherence_measurements else 0.0),
            },
        }

        linked_adapter = self.adapter_engine.create_adapter(
            task_tags=["quantum", "noise", "coherence", "characterization"],
            y_bits=[0] * 16,
            z_bits=[0] * 8,
            x_bits=[0] * 8,
            parameters={
                "experiment_type": "noise_field",
                "coherence_time": float(coherence_time) if isinstance(coherence_time, (int, float)) else 0.0,
                "system_stability": float(statistics.fmean(coherence_measurements) if coherence_measurements else 0.0),
            },
        )

        artifact = QuantumArtifact(
            artifact_id=artifact_id,
            experiment_type="noise_field_scan",
            config=config.__dict__,
            results=results,
            linked_adapter_ids=[linked_adapter.id],
        )
        self._save_artifact(artifact)
        return artifact

    def run_negative_information_experiment(self, config: ExperimentConfig) -> QuantumArtifact:
        results, summary = self._simulate_negative_information(config)

        artifact_id = f"neg_info_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        linked_adapter = self.adapter_engine.create_adapter(
            task_tags=["quantum", "negative_information", "constraint_tracking"],
            y_bits=[0, 0, 1] + [0] * 13,
            z_bits=[1, 0, 0] + [0] * 5,
            x_bits=[1, 1, 0] + [0] * 5,
            parameters=summary,
        )

        artifact = QuantumArtifact(
            artifact_id=artifact_id,
            experiment_type="negative_information_experiment",
            config=config.__dict__,
            results=results,
            linked_adapter_ids=[linked_adapter.id],
        )

        self._save_artifact(artifact)
        return artifact

    def _simulate_negative_information(self, config: ExperimentConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params = config.parameters or {}
        n_qubits = int(params.get("n_qubits", 4))
        n_steps = int(params.get("n_steps", 20))
        exclusion_interval = int(params.get("exclusion_interval", 5))
        exclusion_strength = float(params.get("exclusion_strength", 0.8))
        exclusion_fraction = float(params.get("exclusion_fraction", 0.25))
        evolution_type = str(params.get("evolution_type", "random_walk"))

        dim = 1 << n_qubits

        base_seed = int(config.seed) if config.seed is not None else 0
        # Independent RNG streams so branches are comparable under the same initial seed.
        exclusion_rng = random.Random(base_seed + 17)
        measurement_rng = random.Random(base_seed + 23)

        branch_a = self._run_branch_baseline(dim, n_steps, evolution_type)
        branch_b = self._run_branch_exclusion(
            dim,
            n_steps,
            evolution_type,
            exclusion_interval,
            exclusion_strength,
            exclusion_fraction,
            exclusion_rng,
        )
        branch_c = self._run_branch_measurement(
            dim,
            n_steps,
            evolution_type,
            exclusion_interval,
            measurement_rng,
        )

        metrics = self._compute_branch_metrics(branch_a, branch_b, branch_c)

        results = {
            "branch_a_baseline": branch_a,
            "branch_b_exclusion": branch_b,
            "branch_c_measurement": branch_c,
            "comparative_metrics": metrics,
            "parameters": {
                "n_qubits": n_qubits,
                "n_steps": n_steps,
                "dim": dim,
                "exclusion_interval": exclusion_interval,
                "exclusion_strength": exclusion_strength,
                "exclusion_fraction": exclusion_fraction,
                "evolution_type": evolution_type,
                "seed": base_seed,
            },
            "analysis": self._generate_analysis(metrics),
        }

        adapter_summary = {
            "experiment_type": "negative_information",
            "information_gain_exclusion": metrics.get("info_gain_exclusion", 0.0),
            "information_gain_measurement": metrics.get("info_gain_measurement", 0.0),
            "saturation_point": metrics.get("saturation_point", -1),
        }

        return results, adapter_summary

    def _run_branch_baseline(self, dim: int, n_steps: int, evolution_type: str) -> Dict[str, Any]:
        psi = [0j] * dim
        psi[0] = 1.0 + 0j

        traj: List[Dict[str, Any]] = []
        entropy_history: List[float] = []
        coherence_history: List[float] = []
        prob_history: List[List[float]] = []

        traj.append(self._snapshot(psi, 0))
        prob_history.append(self._probs(psi))
        entropy_history.append(self._shannon_entropy(prob_history[-1]))
        coherence_history.append(self._l1_coherence(psi))

        for step in range(1, n_steps + 1):
            psi = self._evolve_state(psi, evolution_type, step)
            traj.append(self._snapshot(psi, step))
            prob_history.append(self._probs(psi))
            entropy_history.append(self._shannon_entropy(prob_history[-1]))
            coherence_history.append(self._l1_coherence(psi))

        return {
            "final_state_hash": traj[-1]["state_hash"],
            "trajectory": traj,
            "prob_history": prob_history,
            "entropy_history": entropy_history,
            "coherence_history": coherence_history,
            "final_entropy": entropy_history[-1],
            "final_coherence": coherence_history[-1],
            "events": [],
        }

    def _run_branch_exclusion(
        self,
        dim: int,
        n_steps: int,
        evolution_type: str,
        exclusion_interval: int,
        exclusion_strength: float,
        exclusion_fraction: float,
        rng: random.Random,
    ) -> Dict[str, Any]:
        psi = [0j] * dim
        psi[0] = 1.0 + 0j

        traj: List[Dict[str, Any]] = []
        entropy_history: List[float] = []
        coherence_history: List[float] = []
        prob_history: List[List[float]] = []
        exclusion_events: List[Dict[str, Any]] = []

        traj.append(self._snapshot(psi, 0))
        prob_history.append(self._probs(psi))
        entropy_history.append(self._shannon_entropy(prob_history[-1]))
        coherence_history.append(self._l1_coherence(psi))

        for step in range(1, n_steps + 1):
            psi = self._evolve_state(psi, evolution_type, step)

            if exclusion_interval > 0 and (step % exclusion_interval == 0):
                excluded_indices = self._choose_excluded_region(dim, exclusion_fraction, rng)
                psi = self._apply_negative_constraint(psi, excluded_indices, exclusion_strength)

                exclusion_events.append(
                    {
                        "timestamp": step,
                        "excluded_region": {
                            "indices": excluded_indices,
                            "n_excluded": len(excluded_indices),
                            "fraction_excluded": len(excluded_indices) / dim,
                        },
                        "exclusion_strength": exclusion_strength,
                        "state_hash_after": self._state_checksum(psi),
                        "support_size_after": self._support_size(psi),
                        "entropy_after": self._shannon_entropy(self._probs(psi)),
                        "coherence_after": self._l1_coherence(psi),
                    }
                )

            traj.append(self._snapshot(psi, step))
            prob_history.append(self._probs(psi))
            entropy_history.append(self._shannon_entropy(prob_history[-1]))
            coherence_history.append(self._l1_coherence(psi))

        return {
            "final_state_hash": traj[-1]["state_hash"],
            "trajectory": traj,
            "prob_history": prob_history,
            "entropy_history": entropy_history,
            "coherence_history": coherence_history,
            "final_entropy": entropy_history[-1],
            "final_coherence": coherence_history[-1],
            "events": exclusion_events,
            "n_exclusions": len(exclusion_events),
        }

    def _run_branch_measurement(
        self,
        dim: int,
        n_steps: int,
        evolution_type: str,
        measurement_interval: int,
        rng: random.Random,
    ) -> Dict[str, Any]:
        psi = [0j] * dim
        psi[0] = 1.0 + 0j

        traj: List[Dict[str, Any]] = []
        entropy_history: List[float] = []
        coherence_history: List[float] = []
        prob_history: List[List[float]] = []
        measurement_events: List[Dict[str, Any]] = []

        traj.append(self._snapshot(psi, 0))
        prob_history.append(self._probs(psi))
        entropy_history.append(self._shannon_entropy(prob_history[-1]))
        coherence_history.append(self._l1_coherence(psi))

        for step in range(1, n_steps + 1):
            psi = self._evolve_state(psi, evolution_type, step)

            if measurement_interval > 0 and (step % measurement_interval == 0):
                psi, measured_position = self._apply_projective_update(psi, rng)
                measurement_events.append(
                    {
                        "timestamp": step,
                        "measured_position": measured_position,
                        "state_hash_after": self._state_checksum(psi),
                        "support_size_after": self._support_size(psi),
                        "entropy_after": self._shannon_entropy(self._probs(psi)),
                        "coherence_after": self._l1_coherence(psi),
                    }
                )

            traj.append(self._snapshot(psi, step))
            prob_history.append(self._probs(psi))
            entropy_history.append(self._shannon_entropy(prob_history[-1]))
            coherence_history.append(self._l1_coherence(psi))

        return {
            "final_state_hash": traj[-1]["state_hash"],
            "trajectory": traj,
            "prob_history": prob_history,
            "entropy_history": entropy_history,
            "coherence_history": coherence_history,
            "final_entropy": entropy_history[-1],
            "final_coherence": coherence_history[-1],
            "events": measurement_events,
            "n_measurements": len(measurement_events),
        }

    def _evolve_state(self, psi: List[complex], evolution_type: str, step: int) -> List[complex]:
        n = len(psi)

        if evolution_type in {"random_walk", "spectral", "spectral_walk"}:
            psi_k = self._dft_unitary(psi)
            out_k: List[complex] = []
            for k, amp in enumerate(psi_k):
                x = (k - (n / 2.0)) / n
                phase = cmath.exp(-1j * 2.0 * math.pi * 0.75 * (x * x) * (step / max(1, n)))
                out_k.append(amp * phase)
            out = self._idft_unitary(out_k)
            return self._normalize(out)

        if evolution_type == "shift":
            return [psi[(i - 1) % n] for i in range(n)]

        if evolution_type == "phase":
            out = [amp * cmath.exp(1j * 0.3 * step * (i / n)) for i, amp in enumerate(psi)]
            return self._normalize(out)

        return psi

    def _choose_excluded_region(self, dim: int, fraction: float, rng: random.Random) -> List[int]:
        fraction = max(0.0, min(1.0, fraction))
        width = max(1, int(round(dim * fraction)))
        start = rng.randrange(dim)
        return [int((start + i) % dim) for i in range(width)]

    def _apply_negative_constraint(
        self, psi: List[complex], excluded_indices: Sequence[int], strength: float
    ) -> List[complex]:
        strength = max(0.0, min(1.0, strength))
        scale = 0.0 if strength >= 1.0 else (1.0 - strength)
        out = list(psi)
        for idx in excluded_indices:
            out[int(idx)] *= scale
        return self._normalize(out)

    def _apply_projective_update(
        self, psi: List[complex], rng: random.Random
    ) -> Tuple[List[complex], int]:
        probs = self._probs(psi)
        r = rng.random()
        cum = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen = i
                break
        out = [0j] * len(psi)
        out[chosen] = 1.0 + 0j
        return out, chosen

    def _probs(self, psi: Sequence[complex]) -> List[float]:
        probs = [abs(a) ** 2 for a in psi]
        s = sum(probs)
        if s <= 0:
            return [1.0 / len(probs)] * len(probs)
        return [p / s for p in probs]

    def _shannon_entropy(self, probs: Sequence[float]) -> float:
        out = 0.0
        for p in probs:
            if p > 1e-15:
                out -= p * math.log(p, 2)
        return float(out)

    def _support_size(self, psi: Sequence[complex]) -> int:
        probs = self._probs(psi)
        threshold = 0.01 / len(probs)
        return sum(1 for p in probs if p > threshold)

    def _l1_coherence(self, psi: Sequence[complex]) -> float:
        total = sum(abs(a) for a in psi)
        return float(max(0.0, (total * total) - 1.0))

    def _state_checksum(self, psi: Sequence[complex]) -> str:
        h = hashlib.sha256()
        for a in psi:
            h.update(struct.pack("<dd", float(a.real), float(a.imag)))
        return h.hexdigest()

    def _snapshot(self, psi: Sequence[complex], step: int) -> Dict[str, Any]:
        return {
            "step": step,
            "state_hash": self._state_checksum(psi),
            "support_size": self._support_size(psi),
        }

    def _normalize(self, psi: Sequence[complex]) -> List[complex]:
        norm_sq = sum(abs(a) ** 2 for a in psi)
        if norm_sq <= 0:
            n = len(psi)
            return [(1.0 / math.sqrt(n)) + 0j] * n
        inv = 1.0 / math.sqrt(norm_sq)
        return [a * inv for a in psi]

    def _dft_unitary(self, vec: Sequence[complex]) -> List[complex]:
        n = len(vec)
        inv_sqrt = 1.0 / math.sqrt(n)
        out: List[complex] = []
        for k in range(n):
            s = 0j
            for t, amp in enumerate(vec):
                s += amp * cmath.exp(-2j * math.pi * k * t / n)
            out.append(s * inv_sqrt)
        return out

    def _idft_unitary(self, vec: Sequence[complex]) -> List[complex]:
        n = len(vec)
        inv_sqrt = 1.0 / math.sqrt(n)
        out: List[complex] = []
        for t in range(n):
            s = 0j
            for k, amp in enumerate(vec):
                s += amp * cmath.exp(2j * math.pi * k * t / n)
            out.append(s * inv_sqrt)
        return out

    def _compute_branch_metrics(
        self, branch_a: Dict[str, Any], branch_b: Dict[str, Any], branch_c: Dict[str, Any]
    ) -> Dict[str, Any]:
        info_gain_exclusion = branch_a["final_entropy"] - branch_b["final_entropy"]
        info_gain_measurement = branch_a["final_entropy"] - branch_c["final_entropy"]

        entropy_a = branch_a["entropy_history"]
        entropy_b = branch_b["entropy_history"]
        entropy_c = branch_c["entropy_history"]

        saturation_point = -1
        for i in range(5, len(entropy_b)):
            if abs(entropy_b[i] - entropy_b[i - 1]) < 0.01:
                saturation_point = i
                break

        divergence_ab = self._series_l1(entropy_a, entropy_b)
        divergence_ac = self._series_l1(entropy_a, entropy_c)
        divergence_bc = self._series_l1(entropy_b, entropy_c)

        js_ab = self._mean_js(branch_a["prob_history"], branch_b["prob_history"])
        js_ac = self._mean_js(branch_a["prob_history"], branch_c["prob_history"])
        js_bc = self._mean_js(branch_b["prob_history"], branch_c["prob_history"])

        support_a = [t["support_size"] for t in branch_a["trajectory"]]
        support_b = [t["support_size"] for t in branch_b["trajectory"]]
        support_c = [t["support_size"] for t in branch_c["trajectory"]]

        return {
            "info_gain_exclusion": float(info_gain_exclusion),
            "info_gain_measurement": float(info_gain_measurement),
            "exclusion_vs_measurement_ratio": float(
                info_gain_exclusion / (info_gain_measurement + 1e-12)
            ),
            "saturation_point": saturation_point,
            "divergence_baseline_exclusion": float(divergence_ab),
            "divergence_baseline_measurement": float(divergence_ac),
            "divergence_exclusion_measurement": float(divergence_bc),
            "mean_js_baseline_exclusion": float(js_ab),
            "mean_js_baseline_measurement": float(js_ac),
            "mean_js_exclusion_measurement": float(js_bc),
            "final_support_baseline": int(support_a[-1]),
            "final_support_exclusion": int(support_b[-1]),
            "final_support_measurement": int(support_c[-1]),
            "final_coherence_baseline": float(branch_a["final_coherence"]),
            "final_coherence_exclusion": float(branch_b["final_coherence"]),
            "final_coherence_measurement": float(branch_c["final_coherence"]),
        }

    def _series_l1(self, a: Sequence[float], b: Sequence[float]) -> float:
        m = min(len(a), len(b))
        return float(sum(abs(a[i] - b[i]) for i in range(m)))

    def _mean_js(self, p_hist_a: Sequence[Sequence[float]], p_hist_b: Sequence[Sequence[float]]) -> float:
        m = min(len(p_hist_a), len(p_hist_b))
        if m == 0:
            return 0.0
        return float(sum(self._js_divergence(p_hist_a[i], p_hist_b[i]) for i in range(m)) / m)

    def _js_divergence(self, p: Sequence[float], q: Sequence[float]) -> float:
        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
        return 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

    def _kl_divergence(self, p: Sequence[float], q: Sequence[float]) -> float:
        out = 0.0
        for pi, qi in zip(p, q):
            if pi <= 1e-15:
                continue
            out += pi * math.log(pi / max(qi, 1e-15), 2)
        return out

    def _generate_analysis(self, metrics: Dict[str, Any]) -> str:
        lines = [
            "=== Negative Information Experiment Report ===",
            "",
            "What changed (bookkeeping results):",
            f"  - Exclusion branch entropy reduction vs baseline: {metrics['info_gain_exclusion']:.3f} bits",
            f"  - Projective-update branch entropy reduction vs baseline: {metrics['info_gain_measurement']:.3f} bits",
            f"  - Exclusion/measurement information ratio: {metrics['exclusion_vs_measurement_ratio']:.3f}",
            "",
            "What could not change (observed limits):",
            f"  - Exclusion saturation point (Δentropy < 0.01): step {metrics['saturation_point']}",
            "",
            "Information that remains inaccessible under constraints alone:",
            "  - Exclusion updates typically preserve non-zero coherence while narrowing support",
            "  - Projective updates drive coherence toward ~0 and force singleton support",
            "",
            "Interpretation-free comparison signals:",
            f"  - Mean JS divergence (baseline vs exclusion): {metrics['mean_js_baseline_exclusion']:.6f}",
            f"  - Mean JS divergence (baseline vs measurement): {metrics['mean_js_baseline_measurement']:.6f}",
            f"  - Mean JS divergence (exclusion vs measurement): {metrics['mean_js_exclusion_measurement']:.6f}",
        ]
        return "\n".join(lines)

    def _estimate_decay_time(self, xs: Sequence[float], ys: Sequence[float]) -> float | str:
        # Rough exponential fit: y ≈ a exp(-b x) => ln(y) ≈ ln(a) - b x
        pairs: List[Tuple[float, float]] = [(x, y) for x, y in zip(xs, ys) if y > 1e-6]
        if len(pairs) < 2:
            return "N/A"
        x_mean = statistics.fmean(x for x, _ in pairs)
        ylog_mean = statistics.fmean(math.log(y) for _, y in pairs)
        num = sum((x - x_mean) * (math.log(y) - ylog_mean) for x, y in pairs)
        den = sum((x - x_mean) ** 2 for x, _ in pairs)
        if den <= 0:
            return "N/A"
        slope = num / den
        b = -slope
        if b <= 1e-12:
            return "N/A"
        return float(1.0 / b)

    def list_experiments(self) -> List[str]:
        return [
            "interference_experiment",
            "bell_pair_simulation",
            "chsh_test",
            "noise_field_scan",
            "negative_information_experiment",
        ]

    def replay_artifact(self, artifact_id: str) -> Optional[QuantumArtifact]:
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return None

        print(f"Replaying artifact {artifact_id} of type {artifact.experiment_type}")

        if artifact.experiment_type == "negative_information_experiment":
            verification = self._verify_negative_information_artifact(artifact)
            artifact.results["replay_verification"] = verification

        return artifact

    def _verify_negative_information_artifact(self, artifact: QuantumArtifact) -> Dict[str, Any]:
        cfg = artifact.config or {}
        seed = cfg.get("seed")
        params = (cfg.get("parameters") or {}).copy()

        replay_config = ExperimentConfig(
            experiment_type="negative_information_experiment",
            iterations=int(cfg.get("iterations", 0) or 0),
            noise_level=float(cfg.get("noise_level", 0.0) or 0.0),
            seed=int(seed) if seed is not None else None,
            parameters=params,
        )

        replay_results, _ = self._simulate_negative_information(replay_config)

        def _extract_hashes(branch: Dict[str, Any]) -> List[str]:
            return [snap["state_hash"] for snap in branch.get("trajectory", [])]

        mismatches: Dict[str, Any] = {}
        for key in ["branch_a_baseline", "branch_b_exclusion", "branch_c_measurement"]:
            orig = _extract_hashes(artifact.results.get(key, {}))
            rep = _extract_hashes(replay_results.get(key, {}))
            m = min(len(orig), len(rep))
            first_bad = next((i for i in range(m) if orig[i] != rep[i]), None)
            if first_bad is not None or len(orig) != len(rep):
                mismatches[key] = {
                    "first_mismatch_step_index": first_bad,
                    "orig_len": len(orig),
                    "replay_len": len(rep),
                }

        return {"verified": len(mismatches) == 0, "mismatches": mismatches}

    def use_artifact_as_context(self, artifact_id: str, query: str) -> str:
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return f"Artifact {artifact_id} not found"

        context = f"""Quantum Experiment Context:
Type: {artifact.experiment_type}
Configuration: {json.dumps(artifact.config, indent=2)}
Key Results: {json.dumps(artifact.results.get('statistics', {}), indent=2)}

Based on this quantum experiment, respond to: {query}"""
        return context

    def _save_artifact(self, artifact: QuantumArtifact):
        artifact_path = self.artifacts_path / f"{artifact.artifact_id}.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)

        self.experiment_registry["artifacts"].append(artifact.artifact_id)
        self._save_experiment_registry()

    def _load_artifact(self, artifact_id: str) -> Optional[QuantumArtifact]:
        artifact_path = self.artifacts_path / f"{artifact_id}.json"
        if not artifact_path.exists():
            return None
        try:
            with open(artifact_path, "r") as f:
                data = json.load(f)
            return QuantumArtifact(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _generate_histogram(self, data: Sequence[float], bins: int = 50) -> Dict[str, Any]:
        data_list = list(data)
        if not data_list or bins <= 0:
            return {"counts": [], "bin_edges": []}

        mn = min(data_list)
        mx = max(data_list)
        if mx == mn:
            edges = [mn + i for i in range(bins + 1)]
            counts = [0] * bins
            counts[0] = len(data_list)
            return {"counts": counts, "bin_edges": edges}

        width = (mx - mn) / bins
        edges = [mn + i * width for i in range(bins + 1)]
        counts = [0] * bins
        for x in data_list:
            idx = int((x - mn) / width)
            if idx == bins:
                idx = bins - 1
            counts[idx] += 1
        return {"counts": counts, "bin_edges": edges}

    def _generate_correlation_matrix(self, measurements: List[Dict[str, int]]) -> List[List[float]]:
        # Minimal placeholder; keeping API stable.
        return [[1.0, 0.8], [0.8, 1.0]]


__all__ = ["SyntheticQuantumEngine", "ExperimentConfig"]
