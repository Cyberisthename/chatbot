#!/usr/bin/env python3
"""
Synthetic GPU Miner - Main Entry Point

Demonstrates the synthetic GPU mining architecture with adaptive load balancing.
"""

import argparse
import secrets
import signal
import struct
import sys
import time
from typing import Optional

from .device_manager import DeviceManager
from .hash_core import HashCore
from .precompute_cache import PrecomputeCache
from .protocol_layer import MiningJob, ProtocolLayer
from .scheduler import SyntheticGPUScheduler
from .telemetry import TelemetryController


class SyntheticGPUMiner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device_manager = DeviceManager()
        self.protocol = ProtocolLayer()
        self.hash_core = HashCore(max_workers=4)
        self.precompute = PrecomputeCache()
        self.telemetry = TelemetryController(window_size=30)
        devices = self.device_manager.get_all_devices()
        if not devices:
            print("❌ No devices detected. Exiting.")
            sys.exit(1)
        self.scheduler = SyntheticGPUScheduler(
            protocol=self.protocol,
            hash_core=self.hash_core,
            precompute=self.precompute,
            telemetry=self.telemetry,
            devices=devices,
        )
        self.running = False

    def generate_synthetic_job(self, job_id: str, difficulty_bits: int = 24) -> MiningJob:
        prev_block_hash = secrets.token_bytes(32)
        merkle_root = secrets.token_bytes(32)
        version = 1
        timestamp = int(time.time())
        bits = difficulty_bits
        header_prefix = struct.pack('<I', version) + prev_block_hash + merkle_root + struct.pack('<I', timestamp) + struct.pack('<I', bits)
        leading_zeros = difficulty_bits
        target = (1 << (256 - leading_zeros)) - 1
        nonce_start = 0
        nonce_end = 0xFFFFFFFF
        return MiningJob(
            job_id=job_id,
            header_prefix=header_prefix,
            target=target,
            nonce_start=nonce_start,
            nonce_end=nonce_end,
        )

    def start(self, num_jobs: int = 1, difficulty_bits: int = 24, duration: float = 30.0) -> None:
        self.running = True
        print("\n" + "="*80)
        print("🚀 SYNTHETIC GPU MINER - Starting...")
        print("="*80 + "\n")
        print(f"⚙️  Configuration:")
        print(f"   • Number of jobs: {num_jobs}")
        print(f"   • Difficulty: {difficulty_bits} leading zero bits")
        print(f"   • Duration: {duration:.1f}s per job\n")
        devices = self.device_manager.get_all_devices()
        print(f"📊 Detected Devices ({len(devices)}):")
        for device in devices:
            print(f"   • {device.id}: {device.type.value}, perf_score={device.perf_score:.2f}, batch_size={device.batch_size}")
        print("\n" + "="*80 + "\n")
        start_time = time.time()
        for i in range(num_jobs):
            if not self.running:
                break
            job_id = f"job_{i+1}"
            job = self.generate_synthetic_job(job_id, difficulty_bits)
            print(f"📦 Submitting {job_id} (target: {job.target:064x})")
            self.protocol.submit_job(job)
            job_start = time.time()
            while (time.time() - job_start) < duration:
                if not self.running:
                    break
                share = self.protocol.wait_for_share(timeout=0.5)
                if share:
                    print(f"   ✅ Share found! device={share['device_id']}, nonce={share['nonce']:08x}, diff={share['difficulty']:.2f}")
                elapsed = time.time() - job_start
                if elapsed >= 5.0 and int(elapsed) % 5 == 0:
                    self._print_status(elapsed)
            print(f"   ⏱️  Job {job_id} completed ({duration:.1f}s)\n")
        total_time = time.time() - start_time
        print("="*80)
        print(f"✅ Mining session completed in {total_time:.2f}s\n")
        self._print_final_summary()
        print("="*80 + "\n")

    def _print_status(self, elapsed: float) -> None:
        summary = self.telemetry.get_summary()
        hashrate_mh = summary['global_hashrate'] / 1_000_000
        print(f"   📈 Status @ {elapsed:.1f}s: {hashrate_mh:.2f} MH/s, {summary['total_shares']} shares, {summary['total_hashes']:,} hashes")

    def _print_final_summary(self) -> None:
        summary = self.telemetry.get_summary()
        device_summaries = self.telemetry.get_all_device_summaries()
        print("📊 Final Statistics:\n")
        print(f"   Total Hashes:   {summary['total_hashes']:,}")
        print(f"   Total Shares:   {summary['total_shares']}")
        print(f"   Total Errors:   {summary['total_errors']}")
        print(f"   Global Hashrate: {summary['global_hashrate']/1_000_000:.2f} MH/s")
        print(f"   Uptime:         {summary['uptime']:.2f}s\n")
        print("   Device Performance:")
        for dev_sum in device_summaries:
            print(f"      • {dev_sum['device_id']}: {dev_sum['avg_hashrate']/1_000_000:.2f} MH/s, "
                  f"latency={dev_sum['avg_latency']*1000:.1f}ms, errors={dev_sum['error_rate']:.3f}")

    def stop(self) -> None:
        self.running = False
        print("\n🛑 Stopping miner...")
        self.scheduler.shutdown()
        print("✅ Miner stopped.\n")


def main():
    parser = argparse.ArgumentParser(description="Synthetic GPU Miner with Precompute Load Balancer")
    parser.add_argument("--jobs", type=int, default=3, help="Number of mining jobs to run (default: 3)")
    parser.add_argument("--difficulty", type=int, default=20, help="Difficulty in leading zero bits (default: 20)")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration per job in seconds (default: 15.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    miner = SyntheticGPUMiner(verbose=args.verbose)

    def signal_handler(sig, frame):
        miner.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        miner.start(num_jobs=args.jobs, difficulty_bits=args.difficulty, duration=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        miner.stop()


if __name__ == "__main__":
    main()
