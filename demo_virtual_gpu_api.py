#!/usr/bin/env python3
"""
Virtual GPU API Demo

Demonstrates the clean, unified VirtualGPU API for Jarvis-5090X.
"""

from jarvis5090x import (
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    VirtualGPU,
)
from jarvis5090x.types import OperationKind


def main():
    print("=" * 70)
    print("  💎 VIRTUAL GPU API DEMO")
    print("=" * 70)

    devices = [
        AdapterDevice(
            id="api_cpu",
            label="API CPU",
            kind=DeviceKind.CPU,
            perf_score=10.0,
            capabilities={OperationKind.LINALG, OperationKind.GENERIC},
        ),
        AdapterDevice(
            id="api_quantum",
            label="API Quantum",
            kind=DeviceKind.VIRTUAL,
            perf_score=5.0,
            capabilities={OperationKind.QUANTUM},
        ),
    ]

    print("\n✓ Creating Jarvis-5090X in Extreme mode...")
    jarvis = Jarvis5090X.build_extreme(devices)

    print("✓ Initializing VirtualGPU facade...")
    vgpu = VirtualGPU(jarvis)

    print("\n" + "-" * 70)
    print("  Example 1: Matrix-Vector Multiplication")
    print("-" * 70)

    payload = {
        "operation": "matmul",
        "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "vector": [7.0, 8.0, 9.0],
    }

    print("\nPayload:")
    print(f"  matrix: {payload['matrix']}")
    print(f"  vector: {payload['vector']}")

    result = vgpu.submit("linalg", "demo_matmul", payload)
    print(f"\nResult: {result['result']}")

    print("\n" + "-" * 70)
    print("  Example 2: Quantum State Simulation")
    print("-" * 70)

    quantum_payload = {
        "base_state": {"energy": 100, "stability": 1.0},
        "variations": [
            {"energy": 105, "stability": 0.95},
            {"energy": 95, "stability": 1.05},
            {"energy": 102, "stability": 0.98},
        ],
        "scoring_fn": lambda state: state.get("stability", 0) / (abs(state.get("energy", 100) - 100) + 1),
        "top_k": 1,
    }

    print("\nBase state: {energy: 100, stability: 1.0}")
    print("Variations: 3 energy/stability perturbations")

    result = vgpu.submit("quantum", "demo_quantum", quantum_payload)
    print(f"\nCollapsed state: {result['collapsed_state']}")
    print(f"Branch count: {result['branch_count']}")

    print("\n" + "-" * 70)
    print("  Example 3: Benchmark API (Repeat Operations)")
    print("-" * 70)

    bench_payload = {
        "operation": "matmul",
        "matrix": [[2.0, 3.0], [4.0, 5.0]],
        "vector": [6.0, 7.0],
    }

    print("\nRunning operation 5 times...")
    bench_result = vgpu.benchmark("linalg", "demo_bench", bench_payload, repeat=5)

    print(f"Results: {len(bench_result['results'])} runs")
    print(f"First result: {bench_result['results'][0]['result']}")
    print(f"Cache hit rate: {bench_result['stats']['cache']['hit_rate_pct']:.1f}%")
    print(f"Effective TFLOPs: {bench_result['stats']['estimated_tflops']}")

    print("\n" + "-" * 70)
    print("  Example 4: Caching Demonstration")
    print("-" * 70)

    print("\nFirst call (cache miss):")
    result1 = vgpu.submit("linalg", "demo_cache", bench_payload)
    print(f"  Result: {result1['result']}")

    print("\nSecond call (cache hit):")
    result2 = vgpu.submit("linalg", "demo_cache", bench_payload)
    print(f"  Result: {result2['result']}")
    print(f"  Instant return from cache!")

    print("\nThird call (cache hit):")
    result3 = vgpu.submit("linalg", "demo_cache", bench_payload)
    print(f"  Result: {result3['result']}")

    print("\n" + "=" * 70)
    print("  Final Statistics")
    print("=" * 70)

    stats = jarvis.benchmark_stats()
    print(f"  Total operations:       {stats['total_ops']}")
    print(f"  Cache hits:             {stats['cache_hits']}")
    print(f"  Hit rate:               {stats['cache']['hit_rate_pct']:.1f}%")
    print(f"  Backend executions:     {stats['backend_executions']}")
    print(f"  Estimated TFLOPs:       {stats['estimated_tflops']}")
    print(f"  RTX 5090 baseline:      125.0 TFLOPs")
    print(f"  Advantage:              {stats['estimated_tflops'] / 125.0:.2f}x")

    print("\n" + "=" * 70)
    print("  ✅ VirtualGPU API provides clean, unified interface!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
