#!/usr/bin/env python3
"""
Run variations of the Inference-Only experiment to explore parameter space.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from src.core.adapter_engine import AdapterEngine

def run_variation(name, config, adapter_engine, quantum_engine):
    print(f"\n{'=' * 80}")
    print(f"Running variation: {name}")
    print('=' * 80)
    print(f"  n_qubits={config.parameters['n_qubits']}, "
          f"n_steps={config.parameters['n_steps']}, "
          f"inference_interval={config.parameters['inference_interval']}, "
          f"exclusion_strength={config.parameters['exclusion_strength']}, "
          f"exclusion_fraction={config.parameters['exclusion_fraction']}")

    artifact = quantum_engine.run_inference_only_experiment(config)
    metrics = artifact.results["comparative_metrics"]

    print(f"\nResults:")
    print(f"  Information gained: {metrics['info_gain_exclusion']:.6f} bits")
    print(f"  Ratio to measurement: {metrics['exclusion_vs_measurement_ratio']:.2%}")
    print(f"  Threshold (>80%): {'✓ MET' if metrics['threshold_exceeded'] else '✗ NOT MET'}")
    print(f"  Final coherence (exclusion): {metrics['final_coherence_exclusion']:.6f}")
    print(f"  Final coherence (measurement): {metrics['final_coherence_measurement']:.6f}")

    return {
        "name": name,
        "config": config.parameters,
        "results": {
            "info_gain_exclusion": metrics['info_gain_exclusion'],
            "info_gain_measurement": metrics['info_gain_measurement'],
            "ratio": metrics['exclusion_vs_measurement_ratio'],
            "threshold_met": metrics['threshold_exceeded'],
            "final_coherence_exclusion": metrics['final_coherence_exclusion'],
            "final_coherence_measurement": metrics['final_coherence_measurement'],
            "avg_efficiency_exclusion": metrics['avg_efficiency_exclusion'],
            "avg_efficiency_measurement": metrics['avg_efficiency_measurement'],
        },
        "artifact_id": artifact.artifact_id
    }

def main():
    print("=" * 80)
    print("Inference-Only Information Extraction - Parameter Variations")
    print("=" * 80)

    config = {
        "adapters": {"storage_path": "./artifacts/adapters", "auto_create": True},
        "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
    }

    adapter_engine = AdapterEngine(config)
    quantum_engine = SyntheticQuantumEngine(
        artifacts_path="./artifacts/quantum_experiments",
        adapter_engine=adapter_engine
    )

    variations = []

    # Variation 1: Default (already run)
    v1 = ExperimentConfig(
        experiment_type="inference_only_experiment",
        seed=42,
        parameters={
            "n_qubits": 4,
            "n_steps": 30,
            "inference_interval": 5,
            "exclusion_strength": 0.7,
            "exclusion_fraction": 0.2,
            "evolution_type": "random_walk"
        }
    )
    variations.append(run_variation("V1 - Default", v1, adapter_engine, quantum_engine))

    # Variation 2: Stronger exclusion
    v2 = ExperimentConfig(
        experiment_type="inference_only_experiment",
        seed=42,
        parameters={
            "n_qubits": 4,
            "n_steps": 30,
            "inference_interval": 3,
            "exclusion_strength": 0.9,
            "exclusion_fraction": 0.3,
            "evolution_type": "random_walk"
        }
    )
    variations.append(run_variation("V2 - Stronger exclusion", v2, adapter_engine, quantum_engine))

    # Variation 3: More qubits (larger state space)
    v3 = ExperimentConfig(
        experiment_type="inference_only_experiment",
        seed=42,
        parameters={
            "n_qubits": 5,
            "n_steps": 40,
            "inference_interval": 5,
            "exclusion_strength": 0.8,
            "exclusion_fraction": 0.15,
            "evolution_type": "random_walk"
        }
    )
    variations.append(run_variation("V3 - Larger state space", v3, adapter_engine, quantum_engine))

    # Variation 4: More inference steps
    v4 = ExperimentConfig(
        experiment_type="inference_only_experiment",
        seed=42,
        parameters={
            "n_qubits": 4,
            "n_steps": 50,
            "inference_interval": 3,
            "exclusion_strength": 0.85,
            "exclusion_fraction": 0.25,
            "evolution_type": "random_walk"
        }
    )
    variations.append(run_variation("V4 - More inference steps", v4, adapter_engine, quantum_engine))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF VARIATIONS")
    print("=" * 80)
    print()

    for v in variations:
        print(f"{v['name']}:")
        print(f"  Ratio: {v['results']['ratio']:.2%}")
        print(f"  Threshold: {'✓ MET' if v['results']['threshold_met'] else '✗ NOT MET'}")
        print(f"  Coherence preserved: {v['results']['final_coherence_exclusion']:.6f}")
        print()

    # Find best variation
    best = max(variations, key=lambda v: v['results']['ratio'])
    print("=" * 80)
    print(f"BEST RESULT: {best['name']}")
    print("=" * 80)
    print(f"  Ratio to measurement: {best['results']['ratio']:.2%}")
    print(f"  Threshold (>80%): {'✓ MET' if best['results']['threshold_met'] else '✗ NOT MET'}")
    print(f"  Information gained: {best['results']['info_gain_exclusion']:.6f} bits")
    print(f"  Coherence preserved: {best['results']['final_coherence_exclusion']:.6f}")
    print()

    # Save summary
    summary_file = Path("./artifacts/quantum_experiments/inference_variations_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(variations, f, indent=2)

    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
