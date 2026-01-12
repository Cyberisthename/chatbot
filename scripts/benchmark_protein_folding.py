#!/usr/bin/env python3
"""
Benchmark script for protein folding.
Runs various sequences and records performance and energy metrics.
"""

import time
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiversal.multiversal_protein_computer import MultiversalProteinComputer

# Benchmark sequences
BENCHMARK_SEQUENCES = {
    "poly_alanine_10": "AAAAAAAAAA",
    "poly_alanine_20": "AAAAAAAAAAAAAAAAAAAA",
    "mixed_10": "ACDEFGHIKL",
    "mixed_20": "ACDEFGHIKLMNPQRSTVWY",
    "zinc_finger_small": "PYKCPECGKSFSQKSDLVKH", # Partial zinc finger
}

def run_benchmark(n_universes=4, steps=2000):
    computer = MultiversalProteinComputer(log_level=logging.WARNING)
    results = {}
    
    print(f"{'Sequence Name':<25} {'Len':<5} {'Best Energy':<15} {'Runtime (s)':<12} {'E/sec':<10}")
    print("-" * 75)
    
    for name, seq in BENCHMARK_SEQUENCES.items():
        start_time = time.time()
        result = computer.fold_multiversal(
            sequence=seq,
            n_universes=n_universes,
            steps_per_universe=steps,
            save_artifacts=False
        )
        end_time = time.time()
        
        runtime = end_time - start_time
        best_energy = result.best_overall.best_energy
        eps = (n_universes * steps) / runtime
        
        results[name] = {
            "energy": best_energy,
            "runtime": runtime,
            "eps": eps
        }
        
        print(f"{name:<25} {len(seq):<5} {best_energy:<15.4f} {runtime:<12.2f} {eps:<10.1f}")
    
    return results

if __name__ == "__main__":
    print("Running Baseline Benchmark...")
    run_benchmark(n_universes=2, steps=1000)
