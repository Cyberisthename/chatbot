#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis5090x import (
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
    PhaseDetector,
)


def build_phase_dataset(
    num_per_phase: int = 150,
    output_path: str = "data/phase_dataset.json",
    seed: int = 42,
) -> None:
    random.seed(seed)
    
    devices = [
        AdapterDevice(
            id="quantum_0",
            label="Quantum Simulator",
            kind=DeviceKind.VIRTUAL,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.QUANTUM},
        ),
    ]
    orchestrator = Jarvis5090X(devices)
    detector = PhaseDetector(orchestrator)
    
    phase_types = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]
    
    system_sizes = [16, 24, 32, 40]
    depths = [4, 6, 8, 10, 12]
    biases = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    print(f"Building phase dataset with {num_per_phase} samples per phase...")
    print(f"Total experiments: {len(phase_types) * num_per_phase}")
    
    for phase in phase_types:
        print(f"\nGenerating {num_per_phase} samples for phase: {phase}")
        for i in range(num_per_phase):
            system_size = random.choice(system_sizes)
            depth = random.choice(depths)
            bias = random.choice(biases)
            exp_seed = random.randint(1, 100_000)
            
            detector.run_phase_experiment(
                phase_type=phase,
                system_size=system_size,
                depth=depth,
                seed=exp_seed,
                bias=bias,
            )
            
            if (i + 1) % 25 == 0:
                print(f"  Completed {i + 1}/{num_per_phase} samples")
    
    dataset = detector.build_dataset()
    print(f"\nDataset built with {len(dataset)} examples")
    print(f"Saving to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_json(output_path)
    
    print(f"Dataset saved successfully!")
    print(f"\nDataset statistics:")
    phase_counts = {}
    for example in dataset.examples:
        phase_counts[example.phase_label] = phase_counts.get(example.phase_label, 0) + 1
    
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} examples")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Build phase dataset for training")
    parser.add_argument(
        "--num-per-phase",
        type=int,
        default=150,
        help="Number of samples to generate per phase (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/phase_dataset.json",
        help="Output path for the dataset (default: data/phase_dataset.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    build_phase_dataset(
        num_per_phase=args.num_per_phase,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
