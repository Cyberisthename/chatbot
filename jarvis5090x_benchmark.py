#!/usr/bin/env python3
"""
Jarvis-5090X Comprehensive Benchmark Suite

Measures performance with cache cold vs. warm, compression learning,
and effective compute vs. RTX 5090 baseline.
"""

import time
from typing import Any, Dict, List

from jarvis5090x import (
    DEFAULT_CONFIG,
    EXTREME_CONFIG,
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    VirtualGPU,
)
from jarvis5090x.types import OperationKind


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_jarvis_instance(use_extreme: bool = False):
    devices = [
        AdapterDevice(
            id="bench_cpu",
            label="Benchmark CPU",
            kind=DeviceKind.CPU,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.HASHING, OperationKind.LINALG, OperationKind.GENERIC},
        ),
        AdapterDevice(
            id="bench_gpu",
            label="Benchmark GPU",
            kind=DeviceKind.GPU,
            perf_score=500.0,
            max_concurrency=32,
            capabilities={OperationKind.HASHING, OperationKind.LINALG},
        ),
        AdapterDevice(
            id="bench_quantum",
            label="Quantum Adapter",
            kind=DeviceKind.VIRTUAL,
            perf_score=100.0,
            max_concurrency=16,
            capabilities={OperationKind.QUANTUM},
        ),
    ]

    config = EXTREME_CONFIG if use_extreme else DEFAULT_CONFIG
    jarvis = Jarvis5090X(devices, config=config)
    return jarvis


def benchmark_linalg_operations(jarvis: Jarvis5090X, iterations: int = 10000) -> Dict[str, Any]:
    """Benchmark linear algebra operations with cache cold and warm."""
    print(f"\n  Running {iterations:,} linear algebra operations...")

    matrices = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 2.0]],
        [[5.0, 1.0, 0.0], [0.0, 4.0, 2.0], [1.0, 0.0, 3.0]],
    ]
    vectors = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    start_time = time.perf_counter()
    for i in range(iterations):
        matrix_idx = i % len(matrices)
        vector_idx = i % len(vectors)
        payload = {
            "operation": "matmul",
            "matrix": matrices[matrix_idx],
            "vector": vectors[vector_idx],
        }
        jarvis.submit("linalg", f"bench_linalg_{matrix_idx}_{vector_idx}", payload)
    elapsed = time.perf_counter() - start_time

    stats = jarvis.benchmark_stats()
    ops_per_sec = iterations / elapsed
    avg_latency_us = (elapsed / iterations) * 1_000_000

    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "avg_latency_us": avg_latency_us,
        "stats": stats,
    }


def benchmark_mining_operations(jarvis: Jarvis5090X, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark hashing/mining operations."""
    print(f"\n  Running {iterations:,} mining operations...")

    start_time = time.perf_counter()
    for i in range(iterations):
        nonce_base = i * 100
        payload = {
            "data": f"block_header_{i % 10}",
            "nonce": nonce_base,
        }
        jarvis.submit("hashing", f"bench_mining_{i % 10}", payload)
    elapsed = time.perf_counter() - start_time

    stats = jarvis.benchmark_stats()
    ops_per_sec = iterations / elapsed
    avg_latency_us = (elapsed / iterations) * 1_000_000

    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "avg_latency_us": avg_latency_us,
        "stats": stats,
    }


def benchmark_quantum_operations(jarvis: Jarvis5090X, iterations: int = 500) -> Dict[str, Any]:
    """Benchmark quantum approximation operations."""
    print(f"\n  Running {iterations:,} quantum operations...")

    base_states = [
        {"energy": 100, "spin": 0},
        {"energy": 200, "spin": 1},
        {"energy": 150, "spin": -1},
    ]
    variations = [
        [{"energy": 105}, {"energy": 95}],
        [{"energy": 210}, {"energy": 190}],
        [{"energy": 160}, {"energy": 140}],
    ]

    start_time = time.perf_counter()
    for i in range(iterations):
        state_idx = i % len(base_states)
        payload = {
            "base_state": base_states[state_idx],
            "variations": variations[state_idx],
            "scoring_fn": lambda s: 1.0 / (abs(s.get("energy", 100) - 150) + 1),
            "top_k": 1,
        }
        jarvis.submit("quantum", f"bench_quantum_{state_idx}", payload)
    elapsed = time.perf_counter() - start_time

    stats = jarvis.benchmark_stats()
    ops_per_sec = iterations / elapsed
    avg_latency_us = (elapsed / iterations) * 1_000_000

    return {
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "avg_latency_us": avg_latency_us,
        "stats": stats,
    }


def run_cache_warmup_benchmark():
    """Compare performance with cold vs. warm cache."""
    print_header("📊 CACHE WARMUP BENCHMARK")

    jarvis = create_jarvis_instance(use_extreme=False)

    payload = {
        "operation": "matmul",
        "matrix": [[1.0, 2.0], [3.0, 4.0]],
        "vector": [5.0, 6.0],
    }

    print("\n  Cold cache (first 100 ops)...")
    start = time.perf_counter()
    for i in range(100):
        jarvis.submit("linalg", f"warmup_{i % 5}", payload)
    cold_time = time.perf_counter() - start

    print("  Warm cache (next 10,000 ops)...")
    start = time.perf_counter()
    for i in range(10000):
        jarvis.submit("linalg", f"warmup_{i % 5}", payload)
    warm_time = time.perf_counter() - start

    stats = jarvis.benchmark_stats()
    speedup = (cold_time / 100) / (warm_time / 10000)

    print(f"\n  Cold cache: {cold_time:.4f}s for 100 ops ({cold_time/100*1000:.4f} ms/op)")
    print(f"  Warm cache: {warm_time:.4f}s for 10,000 ops ({warm_time/10000*1000:.4f} ms/op)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Cache hit rate: {stats['cache']['hit_rate_pct']:.2f}%")
    print(f"  Effective TFLOPs: {stats['estimated_tflops']}")


def run_standard_benchmark():
    """Run standard benchmark suite."""
    print_header("🏃 STANDARD CONFIGURATION BENCHMARK")

    jarvis = create_jarvis_instance(use_extreme=False)

    linalg_results = benchmark_linalg_operations(jarvis, iterations=10000)
    print(f"  ✓ LinAlg: {linalg_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {linalg_results['avg_latency_us']:.2f} µs")

    mining_results = benchmark_mining_operations(jarvis, iterations=1000)
    print(f"  ✓ Mining: {mining_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {mining_results['avg_latency_us']:.2f} µs")

    quantum_results = benchmark_quantum_operations(jarvis, iterations=500)
    print(f"  ✓ Quantum: {quantum_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {quantum_results['avg_latency_us']:.2f} µs")

    stats = jarvis.benchmark_stats()
    print(f"\n  Final Statistics:")
    print(f"    Total operations: {stats['total_ops']:,}")
    print(f"    Cache hits: {stats['cache_hits']:,} ({stats['cache']['hit_rate_pct']:.1f}%)")
    print(f"    Compressions: {stats['compressions']:,}")
    print(f"    Estimated TFLOPs: {stats['estimated_tflops']}")
    print(f"    RTX 5090 Baseline: 125.0 TFLOPs")
    print(f"    Effective Advantage: {stats['estimated_tflops'] / 125.0:.2f}x")


def run_extreme_benchmark():
    """Run extreme configuration benchmark."""
    print_header("🚀 EXTREME CONFIGURATION BENCHMARK")

    jarvis = create_jarvis_instance(use_extreme=True)
    print("  Using EXTREME_CONFIG:")
    print(f"    - Cache: 1,000,000 items")
    print(f"    - Compression bases: 20,000")
    print(f"    - Quantum branches: 128")

    linalg_results = benchmark_linalg_operations(jarvis, iterations=10000)
    print(f"  ✓ LinAlg: {linalg_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {linalg_results['avg_latency_us']:.2f} µs")

    mining_results = benchmark_mining_operations(jarvis, iterations=1000)
    print(f"  ✓ Mining: {mining_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {mining_results['avg_latency_us']:.2f} µs")

    quantum_results = benchmark_quantum_operations(jarvis, iterations=500)
    print(f"  ✓ Quantum: {quantum_results['ops_per_sec']:.0f} ops/sec")
    print(f"    Avg latency: {quantum_results['avg_latency_us']:.2f} µs")

    stats = jarvis.benchmark_stats()
    print(f"\n  Final Statistics:")
    print(f"    Total operations: {stats['total_ops']:,}")
    print(f"    Cache hits: {stats['cache_hits']:,} ({stats['cache']['hit_rate_pct']:.1f}%)")
    print(f"    Compressions: {stats['compressions']:,}")
    print(f"    Estimated TFLOPs: {stats['estimated_tflops']}")
    print(f"    RTX 5090 Baseline: 125.0 TFLOPs")
    print(f"    Effective Advantage: {stats['estimated_tflops'] / 125.0:.2f}x")


def run_virtual_gpu_api_demo():
    """Demonstrate the VirtualGPU facade API."""
    print_header("💎 VIRTUAL GPU API DEMO")

    jarvis = create_jarvis_instance(use_extreme=False)
    vgpu = VirtualGPU(jarvis)

    print("\n  Using VirtualGPU facade for cleaner API...")

    payload = {
        "operation": "matmul",
        "matrix": [[1.0, 2.0], [3.0, 4.0]],
        "vector": [5.0, 6.0],
    }

    print("  1. Single operation:")
    result = vgpu.submit("linalg", "api_demo_1", payload)
    print(f"     Result: {result['result']}")

    print("\n  2. Benchmark operation (3 runs):")
    bench_result = vgpu.benchmark("linalg", "api_demo_bench", payload, repeat=3)
    print(f"     Runs: {len(bench_result['results'])}")
    print(f"     Stats: Hit rate = {bench_result['stats']['cache']['hit_rate_pct']:.1f}%")

    print("\n  3. Quantum simulation:")
    quantum_payload = {
        "base_state": {"position": 0},
        "variations": [{"position": 1}, {"position": -1}],
        "scoring_fn": lambda s: abs(s.get("position", 0)),
        "top_k": 1,
    }
    quantum_result = vgpu.submit("quantum", "api_demo_quantum", quantum_payload)
    print(f"     Collapsed state: {quantum_result['collapsed_state']}")

    print("\n  ✓ VirtualGPU API provides clean, unified interface")


def main():
    print()
    print("████████████████████████████████████████████████████████████████████████████████")
    print("██                                                                            ██")
    print("██           🎯 JARVIS-5090X COMPREHENSIVE BENCHMARK SUITE 🎯                ██")
    print("██                                                                            ██")
    print("██  Measuring cache performance, compression learning, and effective compute ██")
    print("██                                                                            ██")
    print("████████████████████████████████████████████████████████████████████████████████")

    run_cache_warmup_benchmark()
    run_standard_benchmark()
    run_extreme_benchmark()
    run_virtual_gpu_api_demo()

    print()
    print("=" * 80)
    print("  ✅ BENCHMARK COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
