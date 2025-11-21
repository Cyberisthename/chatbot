#!/usr/bin/env python3
"""
Jarvis-5090X Virtual ASIC Miner Demo

Demonstrates:
- Virtual ASIC device simulation
- Performance modeling with realistic throughput
- Brute-force cryptographic analysis
- Calibration from simulated to real-world metrics
- Mining reward estimation
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
    analyze_mining_performance,
    bruteforce_threshold_bits,
    compute_scale_factor,
    estimate_daily_reward,
    estimate_time_to_crack,
    format_hashrate,
    format_time,
    is_bruteforce_feasible,
    run_mining_job,
    to_real_hashrate,
    combine_hashrates_real,
)
from jarvis5090x.types import OperationKind


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_asic_device() -> AdapterDevice:
    """Create a virtual ASIC mining device."""
    return AdapterDevice(
        id="asic_0",
        label="Virtual ASIC Miner VA-1",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,  # Huge perf score so scheduler prefers it
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "VA-1",
            "hashes_per_second": 1e12,  # 1 TH/s (EXPERIMENTAL SIMULATION)
            "latency_overhead_ms": 0.1,
            "power_watts": 1500,
            "manufacturer": "Virtual Silicon Labs",
        },
    )


def demo_setup() -> Jarvis5090X:
    """Set up Jarvis-5090X with ASIC, GPU, and CPU devices."""
    print_section("🔩 SETUP: Creating Virtual ASIC + GPU Mining Cluster")

    devices = [
        AdapterDevice(
            id="cpu_0",
            label="Mining CPU",
            kind=DeviceKind.CPU,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.HASHING, OperationKind.LINALG},
            metadata={"hashes_per_second": 1e6},  # 1 MH/s
        ),
        AdapterDevice(
            id="gpu_0",
            label="Virtual Mining GPU",
            kind=DeviceKind.GPU,
            perf_score=500.0,
            max_concurrency=32,
            capabilities={OperationKind.HASHING},
            metadata={"hashes_per_second": 1e9},  # 1 GH/s
        ),
        create_asic_device(),
    ]

    print(f"  ✓ Created {len(devices)} mining devices:")
    for device in devices:
        hashrate = device.metadata.get("hashes_per_second", 0)
        print(f"    - {device.label:30s} {format_hashrate(hashrate):>12s}")

    compression = FlopCompressionLayer(max_bases=50, stability_threshold=2)
    cache = InfiniteMemoryCache(max_items=500)
    quantum = QuantumApproximationLayer()
    cluster = AdapterCluster(devices)

    jarvis = Jarvis5090X(devices, compression, cache, quantum, cluster)
    print("  ✓ Jarvis-5090X orchestrator initialized")
    return jarvis


def demo_asic_hashing(jarvis: Jarvis5090X) -> None:
    """Demonstrate ASIC hashing performance."""
    print_section("⚡ DEMO 1: Virtual ASIC Mining Performance")

    payload = {
        "header_prefix": b"\x01" * 76,
        "nonce_start": 0,
        "nonce_count": 1_000_000,  # 1M nonces
        "target": (1 << 224) - 1,
    }

    print(f"  Mining {payload['nonce_count']:,} nonces...")
    result = jarvis.submit("hashing", "asic_demo_1", payload)

    print(f"\n  Results:")
    print(f"    Device:           {result.get('device_id')}")
    print(f"    ASIC Model:       {result.get('asic_model')}")
    print(f"    Hashes Processed: {result.get('hashes_processed'):,}")
    print(f"    Simulated:        {result.get('simulated')}")
    print(f"    Sim Latency:      {result.get('simulated_latency_ms'):.4f} ms")
    print(f"    Effective Rate:   {format_hashrate(result.get('effective_hashrate_hs', 0))}")
    print(f"    Hash:             {result.get('hash')[:32]}...")
    print("  ✓ ASIC simulation complete")


def demo_performance_comparison(jarvis: Jarvis5090X) -> None:
    """Compare ASIC vs GPU vs CPU performance."""
    print_section("📊 DEMO 2: Performance Comparison (ASIC vs GPU vs CPU)")

    workload_sizes = [1_000, 10_000, 100_000]

    print(f"  Running benchmarks with different workload sizes...")
    print(f"  {'Workload':<12s} {'Device':<10s} {'Time (ms)':<12s} {'Hashrate':<12s}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

    for nonce_count in workload_sizes:
        payload = {
            "header_prefix": b"\x01" * 76,
            "nonce_start": 0,
            "nonce_count": nonce_count,
            "target": (1 << 224) - 1,
        }

        # Run job and let scheduler pick device
        start = time.perf_counter()
        result = jarvis.submit("hashing", f"perf_test_{nonce_count}", payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        device_id = result.get("device_id", "unknown")
        hashes = result.get("hashes_processed", nonce_count)
        hashrate = hashes / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        print(f"  {nonce_count:>10,}   {device_id:<10s} {elapsed_ms:>10.4f}   {format_hashrate(hashrate):>12s}")

    print("  ✓ Performance comparison complete")


def demo_brute_force_analysis(jarvis: Jarvis5090X) -> None:
    """Demonstrate brute-force cryptographic analysis."""
    print_section("🔐 DEMO 3: Brute-Force Cryptographic Analysis")

    # Get ASIC hashrate
    asic_hashrate = 1e12  # 1 TH/s

    print(f"  ASIC Hashrate: {format_hashrate(asic_hashrate)}")
    print()
    print(f"  Maximum key sizes that can be cracked in various time windows:")
    print(f"  {'Time Window':<15s} {'Max Bits':<12s} {'Search Space':<20s}")
    print(f"  {'-'*15} {'-'*12} {'-'*20}")

    time_windows = [
        ("1 second", 1),
        ("1 minute", 60),
        ("1 hour", 3600),
        ("1 day", 86400),
        ("1 week", 604800),
        ("1 month", 2592000),
        ("1 year", 31536000),
    ]

    for label, seconds in time_windows:
        max_bits = bruteforce_threshold_bits(asic_hashrate, seconds)
        search_space = 2 ** int(max_bits)
        print(f"  {label:<15s} {max_bits:>11.1f}   {search_space:>19,.0e}")

    print()
    print("  Practical security level analysis:")
    security_levels = [
        ("Weak (40-bit)", 40),
        ("DES (56-bit)", 56),
        ("Medium (64-bit)", 64),
        ("Strong (80-bit)", 80),
        ("AES-128", 128),
        ("AES-256", 256),
    ]

    print(f"  {'Security Level':<20s} {'Bits':<8s} {'Time to Crack':<20s} {'Feasible?':<10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*20} {'-'*10}")

    for label, bits in security_levels:
        time_seconds = estimate_time_to_crack(bits, asic_hashrate)
        time_str = format_time(time_seconds)
        feasible = is_bruteforce_feasible(bits, asic_hashrate, 31536000)  # 1 year
        feasible_str = "YES" if feasible else "NO"
        print(f"  {label:<20s} {bits:<8d} {time_str:<20s} {feasible_str:<10s}")

    print("  ✓ Brute-force analysis complete")


def demo_calibration_and_real_world(jarvis: Jarvis5090X) -> None:
    """Demonstrate calibration from simulated to real-world metrics."""
    print_section("🎯 DEMO 4: Calibration & Real-World Equivalents")

    # Simulated measurements
    H_virtual_sim = 1e12  # 1 TH/s from ASIC simulation
    H_real_measured = 80e6  # 80 MH/s from actual GPU (example)

    print(f"  Calibration Setup:")
    print(f"    Real GPU (measured):      {format_hashrate(H_real_measured)}")
    print(f"    Virtual ASIC (simulated): {format_hashrate(H_virtual_sim)}")
    print()

    # Compute scale factor
    scale_factor = compute_scale_factor(H_real_measured, H_virtual_sim)
    print(f"  Computed Scale Factor: α = {scale_factor:.6e}")
    print()

    # Convert virtual to real
    H_virtual_real = to_real_hashrate(H_virtual_sim, scale_factor)
    print(f"  Virtual ASIC → Real Equivalent: {format_hashrate(H_virtual_real)}")
    print()

    # Combined performance
    H_total = combine_hashrates_real(H_real_measured, H_virtual_sim, scale_factor)
    print(f"  Combined Total Hashrate (Real GPU + Virtual ASIC):")
    print(f"    {format_hashrate(H_total)}")
    print()

    print("  ✓ Calibration complete")


def demo_mining_rewards(jarvis: Jarvis5090X) -> None:
    """Demonstrate mining reward estimation."""
    print_section("💰 DEMO 5: Mining Reward Estimation")

    # Example: Bitcoin-like network
    your_hashrate = 1e12  # 1 TH/s
    network_hashrate = 400e18  # 400 EH/s (realistic for Bitcoin)
    blocks_per_day = 144  # Bitcoin: ~10 min per block
    block_reward = 6.25  # Bitcoin current reward

    print(f"  Mining Scenario (Bitcoin-like network):")
    print(f"    Your hashrate:      {format_hashrate(your_hashrate)}")
    print(f"    Network hashrate:   {format_hashrate(network_hashrate)}")
    print(f"    Blocks per day:     {blocks_per_day}")
    print(f"    Block reward:       {block_reward} BTC")
    print()

    daily_btc = estimate_daily_reward(your_hashrate, network_hashrate, blocks_per_day, block_reward)
    monthly_btc = daily_btc * 30
    yearly_btc = daily_btc * 365

    print(f"  Expected Mining Rewards:")
    print(f"    Daily:   {daily_btc:.8f} BTC")
    print(f"    Monthly: {monthly_btc:.8f} BTC")
    print(f"    Yearly:  {yearly_btc:.8f} BTC")
    print()

    # Show what happens if we increase hashrate 10x
    your_hashrate_10x = your_hashrate * 10
    daily_btc_10x = estimate_daily_reward(your_hashrate_10x, network_hashrate, blocks_per_day, block_reward)

    print(f"  With 10x hashrate ({format_hashrate(your_hashrate_10x)}):")
    print(f"    Daily:   {daily_btc_10x:.8f} BTC")
    print(f"    Monthly: {daily_btc_10x * 30:.8f} BTC")
    print(f"    Yearly:  {daily_btc_10x * 365:.8f} BTC")
    print()

    print("  ✓ Reward estimation complete")


def demo_integrated_analysis(jarvis: Jarvis5090X) -> None:
    """Run integrated performance analysis."""
    print_section("🔬 DEMO 6: Integrated Performance Analysis")

    print("  Running comprehensive performance analysis...")
    print()

    analysis = analyze_mining_performance(jarvis, workload_nonces=100_000)

    print(f"  Analysis Results:")
    print(f"    Workload:         {analysis['workload_nonces']:,} nonces")
    print(f"    Elapsed:          {analysis['elapsed_s']:.4f} seconds")
    print(f"    Hashrate:         {analysis['hashrate_formatted']}")
    print(f"    Device:           {analysis['device_id']}")
    print(f"    Simulated:        {analysis['simulated']}")

    if analysis.get("simulated"):
        print(f"    ASIC Model:       {analysis.get('asic_model')}")
        print(f"    Sim Latency:      {analysis.get('simulated_latency_ms'):.4f} ms")

    print()
    print(f"  Brute-Force Feasibility (max bits for different time windows):")
    bf = analysis["bruteforce_analysis"]
    print(f"    1 second:  {bf['1_second']:.1f} bits")
    print(f"    1 minute:  {bf['1_minute']:.1f} bits")
    print(f"    1 hour:    {bf['1_hour']:.1f} bits")
    print(f"    1 day:     {bf['1_day']:.1f} bits")
    print(f"    1 year:    {bf['1_year']:.1f} bits")
    print()

    print("  ✓ Integrated analysis complete")


def demo_final_stats(jarvis: Jarvis5090X) -> None:
    """Display final statistics."""
    print_section("📈 FINAL STATISTICS")

    stats = jarvis.benchmark_stats()
    print(f"  Total operations:       {stats['total_ops']}")
    print(f"  Cache hits:             {stats['cache_hits']}")
    print(f"  Cache hit rate:         {stats['cache']['hit_rate_pct']:.1f}%")
    print(f"  Backend executions:     {stats['backend_executions']}")
    print(f"  Hashing operations:     {stats['backends']['hashing']}")
    print(f"  Average latency:        {stats['average_latency_ms']:.4f} ms")
    print(f"  Estimated TFLOPs:       {stats['estimated_tflops']}")
    print()
    print("  ✓ Virtual ASIC demonstration complete!")


def main() -> None:
    print()
    print("████████████████████████████████████████████████████████████████████████████████")
    print("██                                                                            ██")
    print("██           🔩 JARVIS-5090X VIRTUAL ASIC MINER DEMO 🔩                      ██")
    print("██                                                                            ██")
    print("██  Simulates ultra-high-performance ASIC mining devices with realistic      ██")
    print("██  performance models, brute-force analysis, and real-world calibration     ██")
    print("██                                                                            ██")
    print("████████████████████████████████████████████████████████████████████████████████")

    jarvis = demo_setup()

    demo_asic_hashing(jarvis)
    demo_performance_comparison(jarvis)
    demo_brute_force_analysis(jarvis)
    demo_calibration_and_real_world(jarvis)
    demo_mining_rewards(jarvis)
    demo_integrated_analysis(jarvis)
    demo_final_stats(jarvis)

    print()
    print("=" * 80)
    print("  ✅ COMPLETE - Virtual ASIC mining simulation ready for research!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
