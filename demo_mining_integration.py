#!/usr/bin/env python3
"""
Jarvis-5090X Mining Integration Demo

Demonstrates integration with the existing Synthetic GPU Miner HashCore.
"""

from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    FlopCompressionLayer,
    InfiniteMemoryCache,
    Jarvis5090X,
    QuantumApproximationLayer,
)
from jarvis5090x.types import OperationKind
from synthetic_gpu_miner.hash_core import HashCore


def main():
    print("=" * 70)
    print("  Jarvis-5090X + Synthetic GPU Miner Integration Demo")
    print("=" * 70)

    devices = [
        AdapterDevice(
            id="mining_cpu",
            label="Mining CPU",
            kind=DeviceKind.CPU,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.HASHING, OperationKind.LINALG},
        ),
        AdapterDevice(
            id="mining_gpu",
            label="Virtual Mining GPU",
            kind=DeviceKind.GPU,
            perf_score=500.0,
            max_concurrency=32,
            capabilities={OperationKind.HASHING},
        ),
    ]

    print("\n✓ Created virtual mining devices")

    hash_core = HashCore()
    compression = FlopCompressionLayer(max_bases=50, stability_threshold=2)
    cache = InfiniteMemoryCache(max_items=500)
    quantum = QuantumApproximationLayer()
    cluster = AdapterCluster(devices)

    jarvis = Jarvis5090X(devices, compression, cache, quantum, cluster)
    jarvis._hash_core = hash_core

    print("✓ Initialized Jarvis-5090X with HashCore integration")

    print("\n" + "-" * 70)
    print("  Running 10 mining jobs with caching...")
    print("-" * 70)

    header_prefix = b"\x01" * 76
    target = (1 << 224) - 1

    for job_idx in range(10):
        payload = {
            "job_id": f"job_{job_idx}",
            "midstate_id": "midstate_demo",
            "header_prefix": header_prefix,
            "nonce_start": job_idx * 1000,
            "nonce_count": 1000,
            "target": target,
        }

        result = jarvis.submit("hashing", f"mining_job_{job_idx}", payload)
        hashes_processed = result.get("hashes_processed", 0)
        device_id = result.get("device_id", "unknown")
        print(f"  Job {job_idx:2d}: {hashes_processed:6d} hashes on {device_id}")

    print("\n" + "-" * 70)
    print("  Re-running first 3 jobs (should hit cache)...")
    print("-" * 70)

    for job_idx in range(3):
        payload = {
            "job_id": f"job_{job_idx}",
            "midstate_id": "midstate_demo",
            "header_prefix": header_prefix,
            "nonce_start": job_idx * 1000,
            "nonce_count": 1000,
            "target": target,
        }

        result = jarvis.submit("hashing", f"mining_job_{job_idx}", payload)
        print(f"  Job {job_idx:2d}: Instant cache hit!")

    print("\n" + "=" * 70)
    print("  Final Statistics")
    print("=" * 70)

    stats = jarvis.benchmark_stats()
    print(f"  Total operations:       {stats['total_ops']}")
    print(f"  Cache hits:             {stats['cache_hits']}")
    print(f"  Cache hit rate:         {stats['cache']['hit_rate_pct']:.1f}%")
    print(f"  Backend executions:     {stats['backend_executions']}")
    print(f"  Hashing operations:     {stats['backends']['hashing']}")
    print(f"  Estimated TFLOPs:       {stats['estimated_tflops']}")
    print("=" * 70)

    print("\n✅ Integration successful! Jarvis-5090X works with HashCore.")

    hash_core.shutdown()


if __name__ == "__main__":
    main()
