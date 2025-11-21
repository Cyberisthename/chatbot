#!/usr/bin/env python3
"""
Quick Start Guide: Virtual ASIC Mining with Jarvis-5090X

This example demonstrates the basics of creating and using virtual ASIC devices.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the Python path when running from the examples directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
    analyze_mining_performance,
    bruteforce_threshold_bits,
    estimate_daily_reward,
    format_hashrate,
    format_time,
)


def main():
    print("=" * 70)
    print("  Virtual ASIC Quick Start")
    print("=" * 70)
    print()

    # Step 1: Create an ASIC device
    print("Step 1: Creating a virtual ASIC device...")
    asic = AdapterDevice(
        id="asic_0",
        label="My First ASIC",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "Quickstart-1",
            "hashes_per_second": 1e12,  # 1 TH/s
            "latency_overhead_ms": 0.1,
        },
    )
    print(f"  ✓ Created: {asic.label}")
    print(f"  ✓ Hashrate: {format_hashrate(asic.metadata['hashes_per_second'])}")
    print()

    # Step 2: Initialize Jarvis
    print("Step 2: Initializing Jarvis-5090X orchestrator...")
    cluster = AdapterCluster([asic])
    jarvis = Jarvis5090X([asic], adapter_cluster=cluster)
    print("  ✓ Orchestrator ready")
    print()

    # Step 3: Submit a mining job
    print("Step 3: Submitting mining job...")
    result = jarvis.submit(
        "hashing",
        "quickstart_job",
        {
            "header_prefix": b"\x01" * 76,
            "nonce_start": 0,
            "nonce_count": 1_000_000,
            "target": (1 << 224) - 1,
        },
    )
    print(f"  ✓ Job complete")
    print(f"  ✓ Hash: {result['hash'][:32]}...")
    print(f"  ✓ Processed: {result['hashes_processed']:,} hashes")
    print(f"  ✓ Latency: {result['simulated_latency_ms']:.4f} ms")
    print()

    # Step 4: Analyze performance
    print("Step 4: Analyzing performance...")
    analysis = analyze_mining_performance(jarvis, workload_nonces=100_000)
    print(f"  ✓ Measured hashrate: {analysis['hashrate_formatted']}")
    print(f"  ✓ Max bits (1 day): {analysis['bruteforce_analysis']['1_day']:.1f}")
    print()

    # Step 5: Estimate mining rewards
    print("Step 5: Estimating mining rewards...")
    daily_btc = estimate_daily_reward(
        your_hashrate_hs=1e12,
        network_hashrate_hs=400e18,  # Bitcoin network
        blocks_per_day=144,
        block_reward_coins=6.25,
    )
    print(f"  ✓ Daily BTC: {daily_btc:.8f}")
    print(f"  ✓ Monthly BTC: {daily_btc * 30:.8f}")
    print()

    # Step 6: Security analysis
    print("Step 6: Security analysis...")
    hashrate = 1e12
    for time_window, seconds in [("1 hour", 3600), ("1 day", 86400), ("1 year", 31536000)]:
        max_bits = bruteforce_threshold_bits(hashrate, seconds)
        print(f"  ✓ Max bits in {time_window}: {max_bits:.1f}")
    print()

    print("=" * 70)
    print("  ✅ Quick start complete!")
    print("  Next: Try demo_asic_mining.py for more examples")
    print("=" * 70)


if __name__ == "__main__":
    main()
