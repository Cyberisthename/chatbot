#!/usr/bin/env python3
"""
Synthetic GPU Miner - Interactive Demo

Demonstrates all key features of the synthetic GPU architecture:
1. Device detection and profiling
2. Adaptive batch sizing
3. Heterogeneous CPU+GPU scheduling
4. Precompute caching and midstates
5. Real-time performance telemetry
"""

import time
from synthetic_gpu_miner.device_manager import DeviceManager
from synthetic_gpu_miner.hash_core import HashCore
from synthetic_gpu_miner.precompute_cache import PrecomputeCache
from synthetic_gpu_miner.protocol_layer import MiningJob, ProtocolLayer
from synthetic_gpu_miner.scheduler import SyntheticGPUScheduler
from synthetic_gpu_miner.telemetry import TelemetryController


def print_banner():
    print("\n" + "="*80)
    print("🎯 SYNTHETIC GPU MINER - Interactive Demo")
    print("="*80 + "\n")


def demo_device_detection():
    print("📊 DEMO 1: Device Detection and Profiling")
    print("-" * 80)
    
    dm = DeviceManager()
    devices = dm.get_all_devices()
    
    print(f"\nDetected {len(devices)} device(s):\n")
    for device in devices:
        print(f"   🔹 {device.id}")
        print(f"      Type:        {device.type.value}")
        print(f"      Perf Score:  {device.perf_score:.2f}")
        print(f"      Batch Size:  {device.batch_size:,} nonces")
        print(f"      Range:       [{device.min_batch_size:,} - {device.max_batch_size:,}]")
        if device.metadata:
            for key, val in device.metadata.items():
                print(f"      {key}:  {val}")
        print()
    
    print("✅ Device detection complete\n")
    input("Press Enter to continue...")
    return dm


def demo_precompute_cache(dm):
    print("\n" + "="*80)
    print("⚡ DEMO 2: Precompute Cache and Midstates")
    print("-" * 80)
    
    cache = PrecomputeCache()
    
    print("\n📝 SHA-256 constants preloaded:")
    print(f"   K table: {len(cache.constants['sha256_k'])} entries")
    print(f"   IV: {len(cache.constants['sha256_iv'])} values")
    
    import secrets
    import struct
    
    print("\n🔄 Computing midstates for 3 different jobs:")
    for i in range(1, 4):
        job_id = f"job_{i}"
        header = secrets.token_bytes(76)
        
        start = time.perf_counter()
        midstate_id = cache.compute_midstate(job_id, header)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"   Job {i}: midstate_id={midstate_id}, time={elapsed:.3f}ms")
    
    print(f"\n💾 Cache contains {len(cache.midstates)} midstate(s)")
    
    print("\n🔄 Testing midstate retrieval:")
    for i in range(1, 4):
        job_id = f"job_{i}"
        header = secrets.token_bytes(76)
        midstate_id = cache.compute_midstate(job_id, header)
        
        start = time.perf_counter()
        payload = cache.get_midstate_payload(midstate_id)
        elapsed = (time.perf_counter() - start) * 1000000
        
        print(f"   Retrieval: {elapsed:.1f}ns (cached)")
    
    print("\n✅ Precompute cache demo complete\n")
    input("Press Enter to continue...")
    return cache


def demo_work_splitting():
    print("\n" + "="*80)
    print("✂️  DEMO 3: Work Unit Splitting and Batching")
    print("-" * 80)
    
    from work_unit import WorkUnit, Batch
    
    print("\n📦 Creating a large work unit:")
    large_unit = WorkUnit(
        job_id="test_job",
        midstate_id="abc123",
        nonce_start=0,
        nonce_count=1_000_000
    )
    print(f"   Original: {large_unit.nonce_count:,} nonces")
    
    print("\n✂️  Splitting into smaller units (max 100K each):")
    parts = large_unit.split(100_000)
    for i, part in enumerate(parts):
        print(f"   Part {i+1}: start={part.nonce_start:,}, count={part.nonce_count:,}")
    
    print(f"\n📊 Created {len(parts)} work units")
    
    print("\n📦 Creating a batch from work units:")
    batch = Batch.from_work_units("cpu_0", parts[:3])
    print(f"   Device:       {batch.device_id}")
    print(f"   Job ID:       {batch.job_id}")
    print(f"   Work Units:   {len(batch.work_units)}")
    print(f"   Total Nonces: {batch.total_nonce_count:,}")
    
    print("\n✅ Work splitting demo complete\n")
    input("Press Enter to continue...")


def demo_adaptive_batching(dm):
    print("\n" + "="*80)
    print("🎚️  DEMO 4: Adaptive Batch Sizing")
    print("-" * 80)
    
    devices = dm.get_all_devices()
    if not devices:
        print("❌ No devices available")
        return
    
    device = devices[0]
    
    print(f"\n🔧 Testing adaptive batch sizing on {device.id}:")
    print(f"   Initial batch size: {device.batch_size:,}")
    
    scenarios = [
        (0.5, "Fast execution (0.5s)"),
        (2.0, "Slow execution (2.0s)"),
        (1.0, "Target latency (1.0s)"),
    ]
    
    target_latency = 1.0
    
    for latency, description in scenarios:
        device.last_latency = latency
        before = device.batch_size
        device.adjust_batch_size(target_latency)
        after = device.batch_size
        change = ((after - before) / before) * 100
        
        print(f"\n   {description}:")
        print(f"      Measured latency: {latency:.2f}s")
        print(f"      Before: {before:,} nonces")
        print(f"      After:  {after:,} nonces")
        print(f"      Change: {change:+.1f}%")
    
    print("\n✅ Adaptive batching demo complete\n")
    input("Press Enter to continue...")


def demo_full_mining_loop(dm, cache):
    print("\n" + "="*80)
    print("⛏️  DEMO 5: Full Mining Loop")
    print("-" * 80)
    
    protocol = ProtocolLayer()
    hash_core = HashCore(max_workers=2)
    telemetry = TelemetryController(window_size=10)
    devices = dm.get_all_devices()
    
    scheduler = SyntheticGPUScheduler(
        protocol=protocol,
        hash_core=hash_core,
        precompute=cache,
        telemetry=telemetry,
        devices=devices
    )
    
    print("\n🎯 Starting mini mining session (15 seconds)...")
    print("   Difficulty: 18 bits")
    print("   Looking for shares...\n")
    
    import secrets
    import struct
    
    job = MiningJob(
        job_id="demo_job",
        header_prefix=secrets.token_bytes(76),
        target=(1 << (256 - 18)) - 1,
        nonce_start=0,
        nonce_end=1 << 24,
    )
    
    protocol.submit_job(job)
    
    start_time = time.time()
    last_status = start_time
    shares_found = 0
    
    while (time.time() - start_time) < 15.0:
        share = protocol.wait_for_share(timeout=0.5)
        if share:
            shares_found += 1
            print(f"   ✅ Share #{shares_found}: nonce={share['nonce']:08x}, "
                  f"device={share['device_id']}, diff={share['difficulty']:.2f}")
        
        if time.time() - last_status >= 3.0:
            summary = telemetry.get_summary()
            hashrate_mh = summary['global_hashrate'] / 1_000_000
            print(f"   📈 Status: {hashrate_mh:.2f} MH/s, "
                  f"{summary['total_hashes']:,} hashes, "
                  f"{summary['total_shares']} shares")
            last_status = time.time()
    
    print("\n📊 Final statistics:")
    summary = telemetry.get_summary()
    print(f"   Total Hashes:    {summary['total_hashes']:,}")
    print(f"   Total Shares:    {summary['total_shares']}")
    print(f"   Global Hashrate: {summary['global_hashrate']/1_000_000:.2f} MH/s")
    print(f"   Duration:        {summary['uptime']:.2f}s")
    
    print("\n   Per-Device Performance:")
    for dev_summary in telemetry.get_all_device_summaries():
        print(f"      {dev_summary['device_id']}: "
              f"{dev_summary['avg_hashrate']/1_000_000:.2f} MH/s, "
              f"latency={dev_summary['avg_latency']*1000:.1f}ms")
    
    scheduler.shutdown()
    print("\n✅ Full mining loop demo complete\n")


def demo_architecture_overview():
    print("\n" + "="*80)
    print("🏗️  DEMO 6: Architecture Overview")
    print("-" * 80)
    
    print("""
    The Synthetic GPU Miner consists of five key layers:
    
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Protocol Layer (Stratum/Pool Communication)             │
    │     • Receives mining jobs from pool                        │
    │     • Submits valid shares                                  │
    │     • Handles job updates and changes                       │
    ├─────────────────────────────────────────────────────────────┤
    │  2. Hash Core Layer (SHA-256 GPU/CPU Implementation)        │
    │     • CPU: Multi-threaded with NumPy vectorization          │
    │     • GPU: CUDA kernels (currently simulated)               │
    │     • Extensible for other algorithms (Scrypt, etc.)        │
    ├─────────────────────────────────────────────────────────────┤
    │  3. Precomputation & Cache Layer (Midstates & Constants)    │
    │     • Computes SHA-256 midstates from fixed header parts    │
    │     • Caches constants (K table, IV)                        │
    │     • Eliminates redundant computation                      │
    ├─────────────────────────────────────────────────────────────┤
    │  4. Synthetic GPU Layer (Intelligent Scheduler)             │
    │     • Maintains logical infinite work queue                 │
    │     • Schedules batches to idle devices                     │
    │     • Adapts batch sizes based on performance               │
    │     • Load balances across heterogeneous devices            │
    ├─────────────────────────────────────────────────────────────┤
    │  5. Control & Telemetry Layer (Performance Monitoring)      │
    │     • Tracks hashrate, latency, error rate per device       │
    │     • Estimates optimal batch sizes                         │
    │     • Triggers adaptive tuning                              │
    │     • Provides real-time statistics                         │
    └─────────────────────────────────────────────────────────────┘
    
    Key Innovations:
    
    ✨ Synthetic Parallelism
       • Logical work queue appears infinite
       • Physical devices pull work on-demand
       • Dynamic task sizing and splitting
    
    ✨ Heterogeneous Computing
       • CPU and GPU treated as unified pool
       • Automatic load balancing
       • Adapts to device capabilities
    
    ✨ Precompute Optimization
       • Midstates computed once, used millions of times
       • Constants preloaded to device memory
       • Minimizes redundant work
    
    ✨ Adaptive Learning
       • Continuously measures performance
       • Auto-tunes batch sizes
       • Self-optimizing behavior
    """)
    
    print("\n✅ Architecture overview complete\n")


def main():
    print_banner()
    
    print("This demo will walk through all major features of the Synthetic GPU Miner.\n")
    input("Press Enter to begin...\n")
    
    dm = demo_device_detection()
    cache = demo_precompute_cache(dm)
    demo_work_splitting()
    demo_adaptive_batching(dm)
    demo_full_mining_loop(dm, cache)
    demo_architecture_overview()
    
    print("="*80)
    print("🎉 Demo Complete!")
    print("="*80)
    print("\nTo run a full mining session:")
    print("  python -m synthetic_gpu_miner.main --jobs 3 --difficulty 20 --duration 15")
    print("\nTo see all options:")
    print("  python -m synthetic_gpu_miner.main --help")
    print("\nFor more information, see synthetic_gpu_miner/README.md\n")


if __name__ == "__main__":
    main()
