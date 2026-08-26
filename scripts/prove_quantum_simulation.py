#!/usr/bin/env python3
"""Generate a concrete proof pack for JARVIS quantum simulation primitives.

This script creates a deterministic JSON artifact that demonstrates:
1. literal complex amplitudes and destructive/constructive interference,
2. multi-qubit state-vector evolution on 4/6/8-qubit registers,
3. 200+ effective quantum-style basis channels via semantic superpositions,
4. braid-matrix / spectral-radius topological metrics, and
5. the integrated transformer -> braid pipeline used by the research engine.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "docs" / "artifacts" / "quantum_simulation_proof" / "quantum_simulation_report.json"
SEED = 42

import sys

sys.path.insert(0, str(ROOT))

from src.core.adapter_engine import AdapterEngine
from src.quantum.synthetic_quantum import ExperimentConfig, SyntheticQuantumEngine
from src.quantum_llm.braid_math import BraidEntropyCalculator, get_braid_metrics
from src.quantum_llm.quantum_attention import QuantumSuperposition, QuantumSuperpositionAttention
from src.research import AutonomousHypothesisEngine


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _complex_preview(amplitudes: Iterable[complex], basis_labels: Iterable[str], limit: int = 8) -> List[Dict[str, Any]]:
    preview = []
    for idx, (amp, label) in enumerate(zip(amplitudes, basis_labels)):
        if idx >= limit:
            break
        preview.append(
            {
                "basis": label,
                "real": float(np.real(amp)),
                "imag": float(np.imag(amp)),
                "probability": float(abs(amp) ** 2),
            }
        )
    return preview


def _build_temp_quantum_engine(work_dir: str) -> SyntheticQuantumEngine:
    adapter_engine = AdapterEngine(
        {
            "adapters": {
                "storage_path": f"{work_dir}/adapters",
                "graph_path": f"{work_dir}/adapter_graph.json",
                "auto_create": True,
            },
            "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8},
        }
    )
    return SyntheticQuantumEngine(
        artifacts_path=f"{work_dir}/artifacts",
        adapter_engine=adapter_engine,
    )


def explicit_state_demo() -> Dict[str, Any]:
    np.random.seed(SEED)

    plus = QuantumSuperposition(
        np.array([1 + 0j, 1 + 0j], dtype=np.complex128),
        ["|0>", "|1>"],
    )
    phase_flipped = QuantumSuperposition(
        np.array([1 + 0j, 1 + 0j], dtype=np.complex128),
        ["|0>", "|1>"],
    )
    phase_flipped.apply_phase_shift(math.pi, basis_idx=1)
    interfered = plus.interfere(phase_flipped, alpha=0.5)

    q_phase_y = QuantumSuperposition(
        np.array([1 + 0j, 1j], dtype=np.complex128),
        ["|0>", "|1>"],
    )
    q_phase_minus_y = QuantumSuperposition(
        np.array([1 + 0j, -1j], dtype=np.complex128),
        ["|0>", "|1>"],
    )
    three_qubit = plus.entangle_with(q_phase_y).entangle_with(q_phase_minus_y)

    return {
        "seed": SEED,
        "one_qubit_destructive_interference": {
            "input_state_a": _complex_preview(plus.amplitudes, plus.basis_labels, limit=2),
            "input_state_b": _complex_preview(phase_flipped.amplitudes, phase_flipped.basis_labels, limit=2),
            "result_state": _complex_preview(interfered.amplitudes, interfered.basis_labels, limit=2),
            "result_probabilities": [float(x) for x in interfered.probabilities()],
            "dominant_basis_after_interference": interfered.basis_labels[int(np.argmax(interfered.probabilities()))],
        },
        "three_qubit_entangled_register": {
            "qubit_count": 3,
            "hilbert_dimension": len(three_qubit.amplitudes),
            "probability_sum": float(np.sum(three_qubit.probabilities())),
            "state_preview": _complex_preview(three_qubit.amplitudes, three_qubit.basis_labels, limit=8),
        },
    }


def multi_qubit_scaling_demo() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis_quantum_proof_") as temp_dir:
        engine = _build_temp_quantum_engine(temp_dir)

        scaling_rows: List[Dict[str, Any]] = []
        for n_qubits in (4, 6, 8):
            dim = 1 << n_qubits
            psi = [0j] * dim
            psi[0] = 1.0 + 0j
            start = time.perf_counter()
            for step in range(1, 5):
                psi = engine._evolve_state(psi, "spectral", step)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            probs = engine._probs(psi)
            scaling_rows.append(
                {
                    "n_qubits": n_qubits,
                    "hilbert_dimension": dim,
                    "complex128_state_bytes": dim * 16,
                    "entropy_bits": float(engine._shannon_entropy(probs)),
                    "support_size": int(engine._support_size(psi)),
                    "state_hash_prefix": engine._state_checksum(psi)[:16],
                    "runtime_ms": round(elapsed_ms, 4),
                    "amplitude_preview": _complex_preview(
                        psi,
                        [f"|{basis:0{n_qubits}b}>" for basis in range(dim)],
                        limit=8,
                    ),
                }
            )

        experiment_config = ExperimentConfig(
            experiment_type="negative_information_experiment",
            iterations=1000,
            noise_level=0.0,
            seed=SEED,
            parameters={
                "n_qubits": 4,
                "n_steps": 30,
                "exclusion_interval": 5,
                "exclusion_strength": 0.8,
                "evolution_type": "random_walk",
            },
        )
        artifact = engine.run_negative_information_experiment(experiment_config)
        replay = engine.replay_artifact(artifact.artifact_id)
        results = artifact.results
        metrics = results["comparative_metrics"]

        return {
            "seed": SEED,
            "literal_qubit_scaling": scaling_rows,
            "negative_information_experiment": {
                "artifact_id": artifact.artifact_id,
                "branch_final_entropies": {
                    "baseline": float(results["branch_a_baseline"]["final_entropy"]),
                    "exclusion": float(results["branch_b_exclusion"]["final_entropy"]),
                    "measurement": float(results["branch_c_measurement"]["final_entropy"]),
                },
                "final_support_sizes": {
                    "baseline": int(metrics["final_support_baseline"]),
                    "exclusion": int(metrics["final_support_exclusion"]),
                    "measurement": int(metrics["final_support_measurement"]),
                },
                "information_gain_ratio": float(metrics["exclusion_vs_measurement_ratio"]),
                "replay_verification": _to_builtin(replay.results.get("replay_verification", {})),
            },
        }


def effective_basis_demo() -> Dict[str, Any]:
    np.random.seed(SEED)
    attention = QuantumSuperpositionAttention(n_basis_states=256, vocab_size=512)

    token_ids = np.array([11, 22, 33, 44], dtype=np.int64)
    query_superpositions = attention.tokens_to_superpositions(token_ids)
    key_superpositions = attention.tokens_to_superpositions(token_ids)
    value_superpositions = attention.tokens_to_superpositions(token_ids)

    for idx, sp in enumerate(query_superpositions):
        sp.apply_phase_shift((idx + 1) * math.pi / 7.0, basis_idx=idx)
    for idx, sp in enumerate(key_superpositions):
        sp.apply_phase_shift((idx + 1) * math.pi / 11.0, basis_idx=255 - idx)

    output, metrics = attention.superposition_attention(
        query_superpositions,
        key_superpositions,
        value_superpositions,
    )
    first_output = np.abs(output[0])
    top_basis = np.argsort(first_output)[-5:][::-1]

    return {
        "seed": SEED,
        "effective_basis_states": attention.n_basis_states,
        "token_count": len(token_ids),
        "output_shape": list(output.shape),
        "query_probability_sum": float(np.sum(query_superpositions[0].probabilities())),
        "top_basis_indices_for_token_0": [int(idx) for idx in top_basis],
        "top_basis_magnitudes_for_token_0": [float(first_output[idx]) for idx in top_basis],
        "first_query_preview": _complex_preview(
            query_superpositions[0].amplitudes,
            [f"basis_{i}" for i in range(attention.n_basis_states)],
            limit=8,
        ),
        "first_token_metrics": _to_builtin(metrics[0]),
    }


def braid_topology_demo() -> Dict[str, Any]:
    calc = BraidEntropyCalculator(n_strands=4)
    trivial_word = [1]
    topological_word = [1, 3, 2] * 6
    inverse_scramble = [1, -1, 2, -2, 3, -3]

    def _summary(word: List[int]) -> Dict[str, Any]:
        matrix = calc.calculate_braid_matrix(word)
        eigenvalues = np.linalg.eigvals(matrix)
        return {
            "word": word,
            "matrix_top_left_2x2": [
                [
                    {"real": float(np.real(matrix[i, j])), "imag": float(np.imag(matrix[i, j]))}
                    for j in range(2)
                ]
                for i in range(2)
            ],
            "spectral_radius": float(max(abs(ev) for ev in eigenvalues)),
            **_to_builtin(get_braid_metrics(word, n_strands=4)),
        }

    return {
        "trivial": _summary(trivial_word),
        "topological": _summary(topological_word),
        "inverse_scramble": _summary(inverse_scramble),
    }


def integrated_research_pipeline_demo() -> Dict[str, Any]:
    np.random.seed(SEED)
    engine = AutonomousHypothesisEngine(inference_cycles=5, random_seed=7)
    observations = engine.load_observations_from_json(ROOT / "demos" / "modern_research_observations.json")
    proposals = engine.propose_hypotheses(observations, top_k=1)
    top = proposals[0]
    metrics = top.evaluation.quantum_metrics

    return {
        "top_candidate_id": top.candidate.id,
        "candidate_type": top.candidate.candidate_type,
        "source_domains": top.candidate.source_domains,
        "overall_score": float(top.evaluation.overall_score),
        "novelty_regime": metrics.novelty_regime,
        "braid_entropy": float(metrics.braid_entropy),
        "braid_word_preview": metrics.braid_word[:12],
        "stable_interference": _to_builtin(metrics.stable_interference.to_dict() if metrics.stable_interference else {}),
        "cycle_metric_preview": _to_builtin([cycle.to_dict() for cycle in metrics.cycle_metrics[:2]]),
        "falsification_controls": top.falsification_plan.controls,
    }


def scaling_notes() -> Dict[str, Any]:
    return {
        "literal_qubit_demo_range": "4-8 literal qubits are shown with full state vectors.",
        "dimension_growth_rule": "Each additional literal qubit doubles the state-vector dimension (2^n).",
        "literal_8_qubits_dimension": 2**8,
        "literal_200_qubits_dimension": str(2**200),
        "effective_channel_demo": {
            "semantic_superposition_basis_states": 256,
            "interpretation": "The repository already supports 256-amplitude semantic superpositions, which satisfies the requested 200+ effective-channel demonstration without claiming a full literal 200-qubit Hilbert-state simulation."
        },
    }


def generate_report(output_path: Path = DEFAULT_OUTPUT_PATH) -> Dict[str, Any]:
    report = {
        "proof_pack": "JARVIS quantum simulation proof",
        "seed": SEED,
        "code_pointers": [
            {
                "path": "src/quantum_llm/quantum_attention.py",
                "lines": "13-84",
                "purpose": "Complex amplitudes, normalization, measurement, tensor-product entanglement, interference, and phase shifts.",
            },
            {
                "path": "src/quantum_llm/quantum_attention.py",
                "lines": "370-467",
                "purpose": "Semantic superposition attention over configurable basis states, used here with 256 effective channels.",
            },
            {
                "path": "src/quantum/synthetic_quantum.py",
                "lines": "294-345, 508-626, 748-792",
                "purpose": "Literal n-qubit state-vector evolution, negative-information experiment, DFT/IDFT unitary propagation, projective updates, and replay verification.",
            },
            {
                "path": "src/quantum_llm/braid_math.py",
                "lines": "12-90",
                "purpose": "Burau braid matrices, spectral-radius entropy, and novelty-regime classification.",
            },
            {
                "path": "src/research/hypothesis_engine.py",
                "lines": "479-535, 699-730",
                "purpose": "Integration from transformer metrics into braid words and stable constructive-interference scoring.",
            },
        ],
        "explicit_state_demo": explicit_state_demo(),
        "multi_qubit_scaling_demo": multi_qubit_scaling_demo(),
        "effective_basis_demo": effective_basis_demo(),
        "braid_topology_demo": braid_topology_demo(),
        "integrated_research_pipeline_demo": integrated_research_pipeline_demo(),
        "scaling_notes": scaling_notes(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_builtin(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = generate_report()
    print(f"Wrote quantum proof report to {DEFAULT_OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "one_qubit_dominant_basis": report["explicit_state_demo"]["one_qubit_destructive_interference"]["dominant_basis_after_interference"],
                "literal_qubit_dimensions": [row["hilbert_dimension"] for row in report["multi_qubit_scaling_demo"]["literal_qubit_scaling"]],
                "effective_basis_states": report["effective_basis_demo"]["effective_basis_states"],
                "topological_regime": report["braid_topology_demo"]["topological"]["novelty_regime"],
                "integrated_top_candidate": report["integrated_research_pipeline_demo"]["top_candidate_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
