#!/usr/bin/env python3
"""
Run negative information tracking experiment
Demonstrates constraint-based quantum state tracking
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from src.core.adapter_engine import AdapterEngine

def main():
    print("=" * 80)
    print("JARVIS-2v Negative Information Experiment")
    print("Constraint-Based Quantum State Tracking")
    print("=" * 80)
    print()
    
    # Initialize engines
    config = {
        "adapters": {"storage_path": "./artifacts/adapters", "auto_create": True},
        "bits": {"y_bits": 16, "z_bits": 8, "x_bits": 8}
    }
    
    adapter_engine = AdapterEngine(config)
    quantum_engine = SyntheticQuantumEngine(
        artifacts_path="./artifacts/quantum_experiments",
        adapter_engine=adapter_engine
    )
    
    # Configure experiment
    experiment_config = ExperimentConfig(
        experiment_type="negative_information_experiment",
        iterations=1000,
        noise_level=0.1,
        seed=42,
        parameters={
            "n_qubits": 4,
            "n_steps": 30,
            "exclusion_interval": 5,
            "exclusion_strength": 0.8,
            "evolution_type": "random_walk"
        }
    )
    
    print("Running experiment with parameters:")
    print(f"  - n_qubits: {experiment_config.parameters['n_qubits']}")
    print(f"  - n_steps: {experiment_config.parameters['n_steps']}")
    print(f"  - exclusion_interval: {experiment_config.parameters['exclusion_interval']}")
    print(f"  - exclusion_strength: {experiment_config.parameters['exclusion_strength']}")
    print(f"  - seed: {experiment_config.seed}")
    print()
    
    # Run experiment
    print("Executing three branches:")
    print("  - Branch A: Baseline (no measurement, no exclusion)")
    print("  - Branch B: Exclusion-only (negative information tracking)")
    print("  - Branch C: Direct position measurement")
    print()
    
    artifact = quantum_engine.run_negative_information_experiment(experiment_config)
    
    print("=" * 80)
    print(f"Experiment completed: {artifact.artifact_id}")
    print("=" * 80)
    print()
    
    # Display results
    results = artifact.results
    metrics = results["comparative_metrics"]
    
    print("RESULTS SUMMARY")
    print("-" * 80)
    print()
    
    print("Branch Final Entropies:")
    print(f"  - Baseline:     {results['branch_a_baseline']['final_entropy']:.3f} bits")
    print(f"  - Exclusion:    {results['branch_b_exclusion']['final_entropy']:.3f} bits")
    print(f"  - Measurement:  {results['branch_c_measurement']['final_entropy']:.3f} bits")
    print()
    
    print("Information Gain:")
    print(f"  - Exclusion:    {metrics['info_gain_exclusion']:.3f} bits")
    print(f"  - Measurement:  {metrics['info_gain_measurement']:.3f} bits")
    print(f"  - Ratio:        {metrics['exclusion_vs_measurement_ratio']:.1%}")
    print()
    
    print("Final Support Sizes:")
    print(f"  - Baseline:     {metrics['final_support_baseline']}")
    print(f"  - Exclusion:    {metrics['final_support_exclusion']}")
    print(f"  - Measurement:  {metrics['final_support_measurement']}")
    print()
    
    print("Saturation Point:")
    print(f"  - Step:         {metrics['saturation_point']}")
    print()
    
    print("Divergence Metrics:")
    print(f"  - Baseline vs Exclusion:    {metrics['divergence_baseline_exclusion']:.3f}")
    print(f"  - Baseline vs Measurement:  {metrics['divergence_baseline_measurement']:.3f}")
    print(f"  - Exclusion vs Measurement: {metrics['divergence_exclusion_measurement']:.3f}")
    print()
    
    print("Exclusion Events:")
    exclusion_events = results['branch_b_exclusion']['events']
    print(f"  - Total exclusions: {len(exclusion_events)}")
    if exclusion_events:
        print(f"  - First exclusion at step {exclusion_events[0]['timestamp']}")
        print(f"    - Excluded {exclusion_events[0]['excluded_region']['n_excluded']} indices")
        print(f"    - Fraction: {exclusion_events[0]['excluded_region']['fraction_excluded']:.1%}")
    print()
    
    print("Measurement Events:")
    measurement_events = results['branch_c_measurement']['events']
    print(f"  - Total measurements: {len(measurement_events)}")
    if measurement_events:
        print(f"  - First measurement at step {measurement_events[0]['timestamp']}")
        print(f"    - Collapsed to position {measurement_events[0]['measured_position']}")
    print()
    
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print(results['analysis'])
    print()
    
    # Save detailed results
    output_file = Path("./artifacts/quantum_experiments") / f"{artifact.artifact_id}_full.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)
    
    print("=" * 80)
    print(f"Full results saved to: {output_file}")
    print(f"Artifact ID: {artifact.artifact_id}")
    print(f"Linked adapter: {artifact.linked_adapter_ids[0]}")
    print()
    print("Replay this experiment with:")
    print(f"  quantum_engine.replay_artifact('{artifact.artifact_id}')")
    print("=" * 80)

if __name__ == "__main__":
    main()
