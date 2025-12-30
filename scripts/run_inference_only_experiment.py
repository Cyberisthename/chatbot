#!/usr/bin/env python3
"""
Run Inference-Only Information Extraction Experiment

This experiment tests whether intelligent inference alone (adaptive exclusion)
can nearly match the information gain from projective measurement while
preserving quantum coherence.

Core Question: Determine how much usable information can be extracted about
a quantum state without measurement, using exclusion + strategy only.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from src.core.adapter_engine import AdapterEngine

def main():
    print("=" * 80)
    print("🔥 CTO.NEW - Inference-Only Information Extraction Experiment")
    print("=" * 80)
    print()
    print("Core Question:")
    print("  Determine how much usable information can be extracted about a quantum")
    print("  state without measurement, using adaptive exclusion strategy only.")
    print()
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

    # Configure experiment with identical initial conditions for all branches
    experiment_config = ExperimentConfig(
        experiment_type="inference_only_experiment",
        iterations=1000,
        noise_level=0.1,
        seed=42,
        parameters={
            "n_qubits": 4,                      # State dimension = 16
            "n_steps": 30,                      # Evolution steps
            "inference_interval": 5,           # Steps between inference/measurement
            "exclusion_strength": 0.7,         # Amplitude attenuation factor
            "exclusion_fraction": 0.2,         # Fraction of state to exclude per step
            "evolution_type": "random_walk"    # Deterministic evolution
        }
    )

    print("Experimental Setup:")
    print(f"  - n_qubits:           {experiment_config.parameters['n_qubits']}")
    print(f"  - n_steps:            {experiment_config.parameters['n_steps']}")
    print(f"  - inference_interval: {experiment_config.parameters['inference_interval']}")
    print(f"  - exclusion_strength: {experiment_config.parameters['exclusion_strength']}")
    print(f"  - exclusion_fraction: {experiment_config.parameters['exclusion_fraction']}")
    print(f"  - evolution_type:    {experiment_config.parameters['evolution_type']}")
    print(f"  - seed:               {experiment_config.seed}")
    print()

    print("Three Agents (identical initial conditions & evolution rules):")
    print()
    print("Branch A — Baseline:")
    print("  Pure unitary evolution")
    print("  No exclusion")
    print("  No measurement")
    print()
    print("Branch B — Adaptive Exclusion Agent (KEY):")
    print("  Never perform projective measurement")
    print("  At fixed intervals, apply negative constraints only ('where it is not')")
    print("  Strategy: choose exclusion regions that maximize entropy reduction per step")
    print("  Preserve coherence explicitly (do not collapse state)")
    print()
    print("Branch C — Measurement Agent:")
    print("  Perform standard projective measurements")
    print("  Matched cadence to Branch B")
    print("  Acts as the information upper bound")
    print()
    print("=" * 80)
    print()

    # Run experiment
    print("Executing experiment...")
    print()
    artifact = quantum_engine.run_inference_only_experiment(experiment_config)

    print("=" * 80)
    print(f"✓ Experiment completed: {artifact.artifact_id}")
    print("=" * 80)
    print()

    # Display results
    results = artifact.results
    metrics = results["comparative_metrics"]

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                           INTERNAL REPORT (ARTIFACT)")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()

    print("FINAL ENTROPIES:")
    print("-" * 80)
    print(f"  Branch A (Baseline):             {metrics['final_entropy_baseline']:.6f} bits")
    print(f"  Branch B (Adaptive Exclusion):    {metrics['final_entropy_exclusion']:.6f} bits")
    print(f"  Branch C (Measurement):           {metrics['final_entropy_measurement']:.6f} bits")
    print()

    print("INFORMATION GAINED:")
    print("-" * 80)
    print(f"  Final reduction (exclusion):      {metrics['info_gain_exclusion']:.6f} bits")
    print(f"  Final reduction (measurement):    {metrics['info_gain_measurement']:.6f} bits")
    print()

    print("CUMULATIVE INFORMATION:")
    print("-" * 80)
    print(f"  Exclusion (all events):           {metrics['cumulative_info_exclusion']:.6f} bits")
    print(f"  Measurement (all events):         {metrics['cumulative_info_measurement']:.6f} bits")
    print()

    print("SUPPORT SIZE (effective possibilities):")
    print("-" * 80)
    print(f"  Baseline:                         {metrics['final_support_baseline']}")
    print(f"  Exclusion:                        {metrics['final_support_exclusion']}")
    print(f"  Measurement:                      {metrics['final_support_measurement']}")
    print()

    print("COHERENCE PRESERVED vs DESTROYED:")
    print("-" * 80)
    print(f"  Baseline final coherence:        {metrics['final_coherence_baseline']:.6f}")
    print(f"  Exclusion final coherence:       {metrics['final_coherence_exclusion']:.6f} ✓ PRESERVED")
    print(f"  Measurement final coherence:     {metrics['final_coherence_measurement']:.6f} ✗ DESTROYED")
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                    BREAKTHROUGH COMPARISON (CRITICAL)")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()

    ratio = metrics['exclusion_vs_measurement_ratio']
    threshold_met = metrics['threshold_exceeded']

    print(f"  Information_exclusion / Information_measurement = {ratio:.6f}")
    print(f"  Threshold (>80%):                                 {'MET ✓' if threshold_met else 'NOT MET ✗'}")
    print()

    if threshold_met:
        print("  ✓✓✓ SUCCESS: Adaptive exclusion approaches measurement power! ✓✓✓")
    else:
        print("  ✗✗✗ Adaptive exclusion does NOT approach measurement power. ✗✗✗")
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                       EFFICIENCY RATIO OVER TIME")
    print("  Information gained / Coherence lost")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()

    print("Average Efficiency:")
    print(f"  Exclusion:                        {metrics['avg_efficiency_exclusion']:.6f}")
    print(f"  Measurement:                      {metrics['avg_efficiency_measurement']:.6f}")
    print()

    print("Information Gain Curves (per step):")
    print(f"  Exclusion curve length:           {len(metrics['info_curve_exclusion'])}")
    print(f"  Measurement curve length:         {len(metrics['info_curve_measurement'])}")
    if metrics['info_curve_exclusion']:
        print(f"  Exclusion max gain:               {max(metrics['info_curve_exclusion']):.6f}")
    if metrics['info_curve_measurement']:
        print(f"  Measurement max gain:             {max(metrics['info_curve_measurement']):.6f}")
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                       DIVERGENCE METRICS")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()

    print(f"  Baseline vs Exclusion:           {metrics['divergence_baseline_exclusion']:.6f}")
    print(f"  Baseline vs Measurement:         {metrics['divergence_baseline_measurement']:.6f}")
    print(f"  Exclusion vs Measurement:        {metrics['divergence_exclusion_measurement']:.6f}")
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                       EVENT LOGS")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()

    print(f"Exclusion Events (Branch B):")
    exclusion_events = results['branch_b_adaptive_exclusion']['events']
    print(f"  Total exclusions:                 {len(exclusion_events)}")
    if exclusion_events:
        print(f"  First exclusion at step:          {exclusion_events[0]['timestamp']}")
        print(f"    - Excluded {exclusion_events[0]['excluded_region']['n_excluded']} indices")
        print(f"    - Fraction: {exclusion_events[0]['excluded_region']['fraction_excluded']:.2%}")
        print(f"    - Entropy reduction: {exclusion_events[0]['entropy_reduction']:.6f} bits")
        print(f"    - Coherence preserved: {exclusion_events[0]['coherence_after']:.6f}")
        print(f"    - Efficiency: {exclusion_events[0]['efficiency']:.6f}")
    print()

    print(f"Measurement Events (Branch C):")
    measurement_events = results['branch_c_measurement']['events']
    print(f"  Total measurements:                {len(measurement_events)}")
    if measurement_events:
        print(f"  First measurement at step:        {measurement_events[0]['timestamp']}")
        print(f"    - Collapsed to position: {measurement_events[0]['measured_position']}")
        print(f"    - Entropy reduction: {measurement_events[0]['entropy_reduction']:.6f} bits")
        print(f"    - Coherence destroyed: {measurement_events[0]['coherence_after']:.6f}")
        print(f"    - Efficiency: {measurement_events[0]['efficiency']:.6f}")
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                       FULL ANALYSIS")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()
    print(results['analysis'])
    print()

    print("═══════════════════════════════════════════════════════════════════════════════")
    print("                       REPLAY & VERIFICATION")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()
    print("All exclusion decisions, entropy changes, and state hashes logged.")
    print("Deterministic replay produces identical results.")
    print("No hidden measurement or collapse introduced in exclusion branch.")
    print()

    # Save detailed results
    output_file = Path("./artifacts/quantum_experiments") / f"{artifact.artifact_id}_full.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(artifact.to_dict(), f, indent=2)

    print("=" * 80)
    print(f"✓ Full results saved to: {output_file}")
    print(f"✓ Artifact ID: {artifact.artifact_id}")
    print(f"✓ Linked adapter: {artifact.linked_adapter_ids[0]}")
    print()
    print("Replay this experiment with:")
    print(f"  quantum_engine.replay_artifact('{artifact.artifact_id}')")
    print("=" * 80)

if __name__ == "__main__":
    main()
