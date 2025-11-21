"""
ASIC Mining Utilities for Jarvis-5090X

Provides tools for:
- Brute-force feasibility calculations
- Hashrate calibration (virtual -> real-world)
- Mining reward estimation
- Performance analysis
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

from .orchestrator import Jarvis5090X


def is_bruteforce_feasible(n_bits: int, hashrate_hs: float, time_seconds: float) -> bool:
    """
    Check if brute-force attack is feasible given computational budget.

    Args:
        n_bits: Size of search space in bits (2^n_bits total possibilities)
        hashrate_hs: Hashrate in hashes per second
        time_seconds: Time window for attack in seconds

    Returns:
        True if 2^n_bits <= hashrate * time, False otherwise
    """
    if hashrate_hs <= 0 or time_seconds <= 0:
        return False
    budget = hashrate_hs * time_seconds
    search_space = 2**n_bits
    return search_space <= budget


def bruteforce_threshold_bits(hashrate_hs: float, time_seconds: float) -> float:
    """
    Calculate maximum key size (in bits) that can be brute-forced.

    Args:
        hashrate_hs: Hashrate in hashes per second
        time_seconds: Time window for attack in seconds

    Returns:
        Maximum n such that 2^n <= hashrate * time
    """
    if hashrate_hs <= 0 or time_seconds <= 0:
        return 0.0
    budget = hashrate_hs * time_seconds
    return math.log2(budget)


def run_mining_job(
    jarvis: Jarvis5090X,
    payload: Dict[str, Any],
    signature: str = "mining_job",
) -> Dict[str, Any]:
    """
    Run a mining job and measure real-world performance.

    Args:
        jarvis: Jarvis-5090X orchestrator instance
        payload: Mining job payload with header_prefix, nonce_start, nonce_count, etc.
        signature: Operation signature for caching

    Returns:
        Dictionary with:
        - result: Mining job result
        - elapsed_s: Real elapsed time
        - hashrate_hs: Measured hashrate (hashes/second)
    """
    start = time.perf_counter()
    result = jarvis.submit("hashing", signature, payload)
    elapsed = time.perf_counter() - start

    hashes = result.get("hashes_processed", payload.get("nonce_count", 0))
    if elapsed <= 0:
        hashrate_hs = 0.0
    else:
        hashrate_hs = hashes / elapsed

    return {
        "result": result,
        "elapsed_s": elapsed,
        "hashrate_hs": hashrate_hs,
    }


def compute_scale_factor(H_real_measured_hs: float, H_real_sim_hs: float) -> float:
    """
    Compute calibration scale factor to convert simulated hashrate to real-world.

    Args:
        H_real_measured_hs: Real-world measured hashrate (from actual hardware)
        H_real_sim_hs: Simulated hashrate from Jarvis-5090X for same workload

    Returns:
        Scale factor alpha = H_real_measured / H_real_sim
    """
    if H_real_sim_hs <= 0:
        return 0.0
    return H_real_measured_hs / H_real_sim_hs


def to_real_hashrate(sim_hashrate_hs: float, scale_factor: float) -> float:
    """
    Convert simulated hashrate to real-world equivalent.

    Args:
        sim_hashrate_hs: Simulated hashrate from Jarvis-5090X
        scale_factor: Calibration factor (from compute_scale_factor)

    Returns:
        Real-world equivalent hashrate
    """
    return sim_hashrate_hs * scale_factor


def combine_hashrates_real(
    H_real_gpu_hs: float,
    H_virtual_sim_hs: float,
    scale_factor: float,
) -> float:
    """
    Combine real GPU hashrate with virtual device hashrate.

    Args:
        H_real_gpu_hs: Real GPU hashrate (measured from actual hardware)
        H_virtual_sim_hs: Virtual device simulated hashrate
        scale_factor: Calibration factor

    Returns:
        Combined total hashrate in real-world terms
    """
    H_virtual_real = to_real_hashrate(H_virtual_sim_hs, scale_factor)
    return H_real_gpu_hs + H_virtual_real


def estimate_daily_reward(
    your_hashrate_hs: float,
    network_hashrate_hs: float,
    blocks_per_day: float,
    block_reward_coins: float,
) -> float:
    """
    Estimate daily mining reward based on hashrate share.

    Args:
        your_hashrate_hs: Your total hashrate
        network_hashrate_hs: Network total hashrate
        blocks_per_day: Expected blocks per day for the network
        block_reward_coins: Reward per block in coins

    Returns:
        Expected coins per day
    """
    if network_hashrate_hs <= 0:
        return 0.0
    share = your_hashrate_hs / network_hashrate_hs
    return share * blocks_per_day * block_reward_coins


def estimate_time_to_crack(
    key_bits: int,
    hashrate_hs: float,
    success_probability: float = 0.5,
) -> float:
    """
    Estimate time required to crack a key of given size.

    Args:
        key_bits: Key size in bits
        hashrate_hs: Available hashrate
        success_probability: Desired probability of success (default 0.5 for 50%)

    Returns:
        Expected time in seconds to achieve success_probability
    """
    if hashrate_hs <= 0:
        return float("inf")
    search_space = 2**key_bits
    expected_attempts = search_space * success_probability
    return expected_attempts / hashrate_hs


def format_hashrate(hashrate_hs: float) -> str:
    """
    Format hashrate with appropriate SI prefix.

    Args:
        hashrate_hs: Hashrate in hashes/second

    Returns:
        Human-readable string (e.g., "1.5 TH/s", "250 MH/s")
    """
    if hashrate_hs >= 1e15:
        return f"{hashrate_hs / 1e15:.2f} PH/s"
    if hashrate_hs >= 1e12:
        return f"{hashrate_hs / 1e12:.2f} TH/s"
    if hashrate_hs >= 1e9:
        return f"{hashrate_hs / 1e9:.2f} GH/s"
    if hashrate_hs >= 1e6:
        return f"{hashrate_hs / 1e6:.2f} MH/s"
    if hashrate_hs >= 1e3:
        return f"{hashrate_hs / 1e3:.2f} KH/s"
    return f"{hashrate_hs:.2f} H/s"


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable string (e.g., "2.5 days", "3 years")
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    if seconds < 31536000:
        return f"{seconds / 86400:.2f} days"
    return f"{seconds / 31536000:.2f} years"


def analyze_mining_performance(
    jarvis: Jarvis5090X,
    workload_nonces: int = 1_000_000,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze mining performance for a given device.

    Args:
        jarvis: Jarvis-5090X orchestrator
        workload_nonces: Number of nonces to test
        device_id: Specific device to test (optional)

    Returns:
        Performance analysis dictionary
    """
    header_prefix = b"\x01" * 76
    payload = {
        "header_prefix": header_prefix,
        "nonce_start": 0,
        "nonce_count": workload_nonces,
        "target": (1 << 224) - 1,
    }

    job_result = run_mining_job(jarvis, payload, f"perf_test_{device_id or 'auto'}")

    result = job_result["result"]
    elapsed = job_result["elapsed_s"]
    hashrate = job_result["hashrate_hs"]

    analysis = {
        "workload_nonces": workload_nonces,
        "elapsed_s": elapsed,
        "hashrate_hs": hashrate,
        "hashrate_formatted": format_hashrate(hashrate),
        "device_id": result.get("device_id"),
        "simulated": result.get("simulated", False),
    }

    if result.get("simulated"):
        analysis["asic_model"] = result.get("asic_model")
        analysis["simulated_latency_ms"] = result.get("simulated_latency_ms")
        analysis["effective_hashrate_hs"] = result.get("effective_hashrate_hs")

    # Add brute-force analysis
    analysis["bruteforce_analysis"] = {
        "1_second": bruteforce_threshold_bits(hashrate, 1),
        "1_minute": bruteforce_threshold_bits(hashrate, 60),
        "1_hour": bruteforce_threshold_bits(hashrate, 3600),
        "1_day": bruteforce_threshold_bits(hashrate, 86400),
        "1_year": bruteforce_threshold_bits(hashrate, 31536000),
    }

    return analysis


__all__ = [
    "is_bruteforce_feasible",
    "bruteforce_threshold_bits",
    "run_mining_job",
    "compute_scale_factor",
    "to_real_hashrate",
    "combine_hashrates_real",
    "estimate_daily_reward",
    "estimate_time_to_crack",
    "format_hashrate",
    "format_time",
    "analyze_mining_performance",
]
