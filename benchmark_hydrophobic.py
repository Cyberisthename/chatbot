#!/usr/bin/env python3
"""
Benchmark script for hydrophobic protein folding.

Tests the ILVVAIL hydrophobic sequence across various configurations
and generates performance metrics.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.multiversal.multiversal_protein_computer import MultiversalProteinComputer


def run_single_benchmark(
    name: str,
    sequence: str,
    n_universes: int,
    steps: int,
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"  Sequence: {sequence}")
    print(f"  Universes: {n_universes}")
    print(f"  Steps: {steps}")
    print(f"{'='*70}")
    
    computer = MultiversalProteinComputer(
        artifacts_dir="./protein_folding_artifacts",
        log_level=40,  # Only show errors
    )
    
    start_time = time.time()
    result = computer.fold_multiversal(
        sequence=sequence,
        n_universes=n_universes,
        steps_per_universe=steps,
        t_start=2.0,
        t_end=0.2,
        base_seed=seed,
        save_artifacts=False,
    )
    elapsed = time.time() - start_time
    
    # Calculate additional metrics
    total_steps = n_universes * steps
    steps_per_second = total_steps / elapsed if elapsed > 0 else 0
    universes_per_second = n_universes / elapsed if elapsed > 0 else 0
    
    benchmark_result = {
        "benchmark_name": name,
        "sequence": sequence,
        "n_universes": n_universes,
        "steps_per_universe": steps,
        "total_steps": total_steps,
        "base_seed": seed,
        "best_energy": result.best_overall.best_energy,
        "energy_mean": result.energy_mean,
        "energy_std": result.energy_std,
        "total_runtime_s": result.total_runtime_s,
        "wallclock_time_s": elapsed,
        "steps_per_second": steps_per_second,
        "universes_per_second": universes_per_second,
        "acceptance_rates": [u.acceptance_rate for u in result.universes],
        "mean_acceptance_rate": sum(u.acceptance_rate for u in result.universes) / len(result.universes),
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n✓ Complete!")
    print(f"  Best Energy:  {benchmark_result['best_energy']:.6f}")
    print(f"  Runtime:      {elapsed:.3f}s")
    print(f"  Throughput:   {steps_per_second:.0f} steps/s")
    
    return benchmark_result


def main():
    hydrophobic_sequence = "ILVVAIL"
    
    print("="*70)
    print("  HYDROPHOBIC SEQUENCE PROTEIN FOLDING BENCHMARK")
    print("="*70)
    print(f"Sequence: {hydrophobic_sequence}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Define benchmark configurations
    benchmarks = [
        {
            "name": "quick_test",
            "n_universes": 2,
            "steps": 500,
            "description": "Quick validation (2 universes, 500 steps)"
        },
        {
            "name": "baseline",
            "n_universes": 4,
            "steps": 1000,
            "description": "Baseline (4 universes, 1000 steps)"
        },
        {
            "name": "standard",
            "n_universes": 6,
            "steps": 2000,
            "description": "Standard (6 universes, 2000 steps)"
        },
        {
            "name": "high_parallel",
            "n_universes": 8,
            "steps": 2000,
            "description": "High parallelism (8 universes, 2000 steps)"
        },
        {
            "name": "deep_search",
            "n_universes": 4,
            "steps": 5000,
            "description": "Deep search (4 universes, 5000 steps)"
        },
        {
            "name": "intensive",
            "n_universes": 6,
            "steps": 5000,
            "description": "Intensive (6 universes, 5000 steps)"
        },
    ]
    
    results = []
    total_start = time.time()
    
    for i, config in enumerate(benchmarks, 1):
        print(f"\nBenchmark {i}/{len(benchmarks)}: {config['description']}")
        
        result = run_single_benchmark(
            name=config["name"],
            sequence=hydrophobic_sequence,
            n_universes=config["n_universes"],
            steps=config["steps"],
            seed=42,
        )
        result["description"] = config["description"]
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Generate benchmark report
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Configuration':<20} {'Best E':<12} {'Runtime (s)':<12} {'Steps/s':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['benchmark_name']:<20} "
              f"{r['best_energy']:<12.6f} "
              f"{r['wallclock_time_s']:<12.3f} "
              f"{r['steps_per_second']:<12.0f}")
    
    # Performance analysis
    print("\n" + "="*70)
    print("  PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Find best energy
    best_result = min(results, key=lambda x: x["best_energy"])
    print(f"\n✓ Best Energy Found:")
    print(f"  Configuration: {best_result['benchmark_name']} ({best_result['description']})")
    print(f"  Energy: {best_result['best_energy']:.6f}")
    
    # Find fastest
    fastest_result = min(results, key=lambda x: x["wallclock_time_s"])
    print(f"\n✓ Fastest Configuration:")
    print(f"  Configuration: {fastest_result['benchmark_name']} ({fastest_result['description']})")
    print(f"  Runtime: {fastest_result['wallclock_time_s']:.3f}s")
    
    # Find highest throughput
    throughput_result = max(results, key=lambda x: x["steps_per_second"])
    print(f"\n✓ Highest Throughput:")
    print(f"  Configuration: {throughput_result['benchmark_name']} ({throughput_result['description']})")
    print(f"  Throughput: {throughput_result['steps_per_second']:.0f} steps/s")
    
    # Scaling analysis
    print("\n" + "-"*70)
    print("  SCALING ANALYSIS (Fixed 2000 steps)")
    print("-"*70)
    
    scaling_results = [r for r in results if r["steps_per_universe"] == 2000]
    if len(scaling_results) >= 2:
        print(f"\n{'Universes':<12} {'Runtime (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
        print("-"*50)
        
        baseline_time = scaling_results[0]["wallclock_time_s"]
        baseline_universes = scaling_results[0]["n_universes"]
        
        for r in scaling_results:
            speedup = baseline_time / r["wallclock_time_s"]
            efficiency = (speedup / (r["n_universes"] / baseline_universes)) * 100
            print(f"{r['n_universes']:<12} "
                  f"{r['wallclock_time_s']:<12.3f} "
                  f"{speedup:<12.2f} "
                  f"{efficiency:<12.1f}%")
    
    # Save benchmark results
    report = {
        "sequence": hydrophobic_sequence,
        "benchmark_timestamp": datetime.now().isoformat(),
        "total_benchmark_time_s": total_elapsed,
        "num_configurations": len(results),
        "results": results,
    }
    
    report_path = Path("./protein_folding_artifacts/benchmark_hydrophobic.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ Benchmark complete!")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Report saved to: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
