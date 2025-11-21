#!/usr/bin/env python3
"""
Test ASIC simulation functionality.
"""

from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    analyze_mining_performance,
    bruteforce_threshold_bits,
    estimate_time_to_crack,
    format_hashrate,
    format_time,
    is_bruteforce_feasible,
)
from jarvis5090x.types import OperationKind


def test_asic_device_creation():
    """Test that ASIC device can be created and configured."""
    asic = AdapterDevice(
        id="asic_test",
        label="Test ASIC",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "TEST-1",
            "hashes_per_second": 1e12,
            "latency_overhead_ms": 0.1,
        },
    )

    assert asic.id == "asic_test"
    assert asic.kind == DeviceKind.VIRTUAL
    assert asic.metadata["asic_model"] == "TEST-1"
    assert asic.metadata["hashes_per_second"] == 1e12
    print("✓ ASIC device creation test passed")


def test_asic_hashing():
    """Test that ASIC device can process hashing workloads."""
    asic = AdapterDevice(
        id="asic_0",
        label="Test ASIC",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "TEST-1",
            "hashes_per_second": 1e9,  # 1 GH/s for testing
            "latency_overhead_ms": 0.1,
        },
    )

    cluster = AdapterCluster([asic])
    jarvis = Jarvis5090X([asic], adapter_cluster=cluster)

    payload = {
        "header_prefix": b"\x01" * 76,
        "nonce_start": 0,
        "nonce_count": 1000,
        "target": (1 << 224) - 1,
    }

    result = jarvis.submit("hashing", "test_asic_hash", payload)

    assert result.get("device_id") == "asic_0"
    assert result.get("simulated") is True
    assert result.get("asic_model") == "TEST-1"
    assert result.get("hashes_processed") == 1000
    assert result.get("effective_hashrate_hs") == 1e9
    assert "hash" in result
    assert "simulated_latency_ms" in result

    print("✓ ASIC hashing test passed")


def test_bruteforce_analysis():
    """Test brute-force analysis functions."""
    hashrate = 1e12  # 1 TH/s

    # Test feasibility check
    assert is_bruteforce_feasible(10, hashrate, 1) is True  # 2^10 < 1e12
    assert is_bruteforce_feasible(50, hashrate, 1) is False  # 2^50 > 1e12

    # Test threshold calculation
    threshold_1s = bruteforce_threshold_bits(hashrate, 1)
    assert 39 < threshold_1s < 41  # log2(1e12) ≈ 39.86

    threshold_1d = bruteforce_threshold_bits(hashrate, 86400)
    assert 55 < threshold_1d < 57  # log2(1e12 * 86400) ≈ 56.13

    # Test time to crack
    time_40bits = estimate_time_to_crack(40, hashrate)
    assert time_40bits < 2  # 2^40 / 1e12 ≈ 1.1 seconds for 50% probability

    print("✓ Brute-force analysis test passed")


def test_format_functions():
    """Test formatting utility functions."""
    # Test hashrate formatting
    assert "1.00 TH/s" in format_hashrate(1e12)
    assert "1.00 GH/s" in format_hashrate(1e9)
    assert "1.00 MH/s" in format_hashrate(1e6)

    # Test time formatting
    assert "seconds" in format_time(30)
    assert "minutes" in format_time(120)
    assert "hours" in format_time(7200)
    assert "days" in format_time(172800)

    print("✓ Format functions test passed")


def test_integrated_analysis():
    """Test integrated performance analysis."""
    asic = AdapterDevice(
        id="asic_0",
        label="Test ASIC",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "TEST-1",
            "hashes_per_second": 1e9,
            "latency_overhead_ms": 0.1,
        },
    )

    cluster = AdapterCluster([asic])
    jarvis = Jarvis5090X([asic], adapter_cluster=cluster)

    analysis = analyze_mining_performance(jarvis, workload_nonces=10_000)

    assert analysis["workload_nonces"] == 10_000
    assert "hashrate_hs" in analysis
    assert "hashrate_formatted" in analysis
    assert analysis["device_id"] == "asic_0"
    assert analysis["simulated"] is True
    assert "bruteforce_analysis" in analysis
    assert "1_day" in analysis["bruteforce_analysis"]

    print("✓ Integrated analysis test passed")


def main():
    print("=" * 70)
    print("  Testing ASIC Simulation")
    print("=" * 70)
    print()

    test_asic_device_creation()
    test_asic_hashing()
    test_bruteforce_analysis()
    test_format_functions()
    test_integrated_analysis()

    print()
    print("=" * 70)
    print("  ✅ All ASIC simulation tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
