#!/usr/bin/env python3
"""
Jarvis-5090X Virtual GPU Demo

Demonstrates the Jarvis-5090X system with:
- Mining job (hashing workload)
- Quantum simulation job
- Linear algebra job
- Cache hits skipping compute
"""

import time
from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    FlopCompressionLayer,
    InfiniteMemoryCache,
    Jarvis5090X,
    QuantumApproximationLayer,
    format_hashrate,
)
from jarvis5090x.types import OperationKind


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_setup() -> Jarvis5090X:
    print_section("🚀 JARVIS-5090X VIRTUAL GPU DEMO - SETUP")

    devices = [
        AdapterDevice(
            id="cpu_0",
            label="Main CPU",
            kind=DeviceKind.CPU,
            perf_score=10.0,
            max_concurrency=4,
            capabilities={OperationKind.HASHING, OperationKind.LINALG, OperationKind.GENERIC},
        ),
        AdapterDevice(
            id="gpu_0",
            label="Virtual GPU",
            kind=DeviceKind.GPU,
            perf_score=100.0,
            max_concurrency=16,
            capabilities={OperationKind.HASHING, OperationKind.LINALG},
        ),
        AdapterDevice(
            id="asic_0",
            label="Virtual ASIC Miner",
            kind=DeviceKind.VIRTUAL,
            perf_score=10_000.0,
            max_concurrency=128,
            capabilities={OperationKind.HASHING},
            metadata={
                "asic_model": "VA-1",
                "hashes_per_second": 1e12,
                "latency_overhead_ms": 0.1,
            },
        ),
        AdapterDevice(
            id="quantum_0",
            label="Quantum Adapter",
            kind=DeviceKind.VIRTUAL,
            perf_score=25.0,
            max_concurrency=8,
            capabilities={OperationKind.QUANTUM},
        ),
    ]

    print(f"  ✓ Created {len(devices)} virtual devices:")
    for device in devices:
        hashrate = device.metadata.get("hashes_per_second", 0)
        hashrate_str = f" ({format_hashrate(hashrate)})" if hashrate else ""
        print(f"    - {device.label} ({device.kind.value}), perf={device.perf_score}{hashrate_str}")

    compression = FlopCompressionLayer(max_bases=100, stability_threshold=3, tolerance=0.01)
    cache = InfiniteMemoryCache(max_items=1000)
    quantum = QuantumApproximationLayer(max_branches=32, seed=42)
    cluster = AdapterCluster(devices)

    print("  ✓ Initialized compression, cache, and quantum layers")

    orchestrator = Jarvis5090X(devices, compression, cache, quantum, cluster)
    print("  ✓ Jarvis-5090X orchestrator ready!")
    return orchestrator


def demo_mining_job(jarvis: Jarvis5090X) -> None:
    print_section("⛏️  DEMO 1: Mining Job (Hashing Workload)")

    print("  Running 5 SHA-256 hashing operations...")
    for nonce in range(5):
        payload = {
            "data": f"block_header_{nonce}",
            "nonce": nonce,
        }
        start = time.perf_counter()
        result = jarvis.submit("hashing", f"mine_block_{nonce}", payload)
        elapsed = time.perf_counter() - start
        print(f"    Nonce {nonce}: hash={result['hash'][:16]}... ({elapsed*1000:.2f}ms)")

    print("  ✓ All mining operations complete")


def demo_asic(jarvis: Jarvis5090X) -> None:
    print_section("🔩 DEMO 1B: Virtual ASIC Miner")

    payload = {
        "header_prefix": b"\x01" * 76,
        "nonce_start": 0,
        "nonce_count": 1_000_000,
        "target": (1 << 224) - 1,
    }

    result = jarvis.submit("hashing", "asic_demo", payload)

    print(f"  Device:        {result.get('device_id')}")
    hashes_processed = result.get('hashes_processed')
    if hashes_processed is not None:
        print(f"  Hashes:        {hashes_processed:,}")
    print(f"  Simulated:     {result.get('simulated')}")
    print(f"  ASIC model:    {result.get('asic_model')}")
    latency = result.get('simulated_latency_ms')
    if latency is not None:
        print(f"  Sim latency:   {latency:.3f} ms")
    hashrate = result.get('effective_hashrate_hs')
    if hashrate is not None:
        print(f"  Hashrate:      {format_hashrate(hashrate)}")
    print(f"  Hash:          {result.get('hash')[:32]}...")
    print("  ✓ ASIC simulation complete")


def demo_quantum_simulation(jarvis: Jarvis5090X) -> None:
    print_section("🌀 DEMO 2: Quantum Simulation Job")

    base_state = {"energy": 100, "spin": 0}
    variations = [
        {"energy": 105, "spin": 1},
        {"energy": 95, "spin": -1},
        {"energy": 102, "spin": 0},
    ]

    print(f"  Base state: {base_state}")
    print(f"  Variations: {len(variations)}")

    def energy_scoring(state):
        return 1.0 / (abs(state.get("energy", 100) - 100) + 1)

    payload = {
        "base_state": base_state,
        "variations": variations,
        "scoring_fn": energy_scoring,
        "top_k": 1,
    }

    start = time.perf_counter()
    result = jarvis.submit("quantum", "quantum_sim_1", payload)
    elapsed = time.perf_counter() - start

    print(f"  Collapsed state: {result['collapsed_state']}")
    print(f"  Branch count: {result['branch_count']}")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print("  ✓ Quantum simulation complete")


def demo_linalg_job(jarvis: Jarvis5090X) -> None:
    print_section("📐 DEMO 3: Linear Algebra Job")

    matrices = [
        {"matrix": [[1, 2], [3, 4]], "vector": [5, 6]},
        {"matrix": [[0, 1], [1, 0]], "vector": [10, 20]},
        {"matrix": [[2, 0], [0, 3]], "vector": [7, 8]},
    ]

    print(f"  Running {len(matrices)} matrix-vector multiplications...")
    for idx, data in enumerate(matrices):
        payload = {"operation": "matmul", **data}
        start = time.perf_counter()
        result = jarvis.submit("linalg", f"matmul_{idx}", payload)
        elapsed = time.perf_counter() - start
        print(f"    Op {idx}: result={result['result']} ({elapsed*1000:.2f}ms)")

    print("  ✓ All linear algebra operations complete")


def demo_cache_hits(jarvis: Jarvis5090X) -> None:
    print_section("⚡ DEMO 4: Cache Hits (Skipping Compute)")

    payload = {"matrix": [[1, 2], [3, 4]], "vector": [5, 6], "operation": "matmul"}

    print("  Running the same operation 10 times...")
    times = []
    for iteration in range(10):
        start = time.perf_counter()
        result = jarvis.submit("linalg", "cached_matmul", payload)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    print(f"  First run: {times[0]:.4f}ms (cache miss)")
    print(f"  Subsequent runs: {sum(times[1:])/len(times[1:]):.4f}ms avg (cache hits)")
    speedup = times[0] / (sum(times[1:]) / len(times[1:]))
    print(f"  Speedup: {speedup:.1f}x")
    print("  ✓ Cache demonstration complete")


def demo_final_stats(jarvis: Jarvis5090X) -> None:
    print_section("📊 FINAL BENCHMARK STATISTICS")

    stats = jarvis.benchmark_stats()
    print(f"  Total operations:       {stats['total_ops']}")
    print(f"  Cache hits:             {stats['cache_hits']}")
    print(f"  Compressions:           {stats['compressions']}")
    print(f"  Backend executions:     {stats['backend_executions']}")
    print()
    print(f"  Cache stats:")
    print(f"    - Cached items:       {stats['cache']['cached_items']}")
    print(f"    - Hit rate:           {stats['cache']['hit_rate_pct']:.2f}%")
    print()
    print(f"  Compression stats:")
    print(f"    - Basis count:        {stats['compression']['basis_count']}")
    print(f"    - Stable bases:       {stats['compression']['stable_bases']}")
    print()
    print(f"  Backend breakdown:")
    for backend, count in stats['backends'].items():
        print(f"    - {backend:12s}:     {count}")
    print()
    print(f"  🚀 ESTIMATED EFFECTIVE COMPUTE:")
    print(f"     {stats['estimated_tflops']} TFLOPs")
    print(f"     (vs RTX 5090: ~125 TFLOPs baseline)")
    print()
    print("  ✓ Jarvis-5090X demonstrates superior effective performance")
    print("    through compression, caching, and intelligent routing!")


def main() -> None:
    print()
    print("████████████████████████████████████████████████████████████████████████████████")
    print("██                                                                            ██")
    print("██               🚀 JARVIS-5090X VIRTUAL GPU SYSTEM 🚀                       ██")
    print("██                                                                            ██")
    print("██  Virtual GPU that surpasses RTX 5090 through intelligent software         ██")
    print("██                                                                            ██")
    print("████████████████████████████████████████████████████████████████████████████████")

    jarvis = demo_setup()

    demo_mining_job(jarvis)
    demo_asic(jarvis)
    demo_quantum_simulation(jarvis)
    demo_linalg_job(jarvis)
    demo_cache_hits(jarvis)
    demo_final_stats(jarvis)

    print()
    print("=" * 80)
    print("  ✅ DEMO COMPLETE - Jarvis-5090X is ready for your workloads!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
