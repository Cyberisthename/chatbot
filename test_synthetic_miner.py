#!/usr/bin/env python3
"""
Test Suite for Synthetic GPU Miner

Tests all major components and integration.
"""

import time
import secrets
import struct
from synthetic_gpu_miner import (
    WorkUnit, Batch, Device, DeviceType,
    DeviceManager, HashCore, PrecomputeCache,
    ProtocolLayer, SyntheticGPUScheduler, TelemetryController,
)
from synthetic_gpu_miner.protocol_layer import MiningJob


def test_work_unit():
    print("Testing WorkUnit...")
    unit = WorkUnit(job_id="test_job", midstate_id="abc123", nonce_start=0, nonce_count=1000)
    
    # Test splitting
    parts = unit.split(300)
    assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}"
    assert sum(p.nonce_count for p in parts) == 1000, "Total nonce count mismatch"
    
    # Test no split needed
    parts2 = unit.split(2000)
    assert len(parts2) == 1, "Should not split when max_count > nonce_count"
    
    print("✅ WorkUnit tests passed")


def test_batch():
    print("Testing Batch...")
    units = [
        WorkUnit("job1", "mid1", 0, 100),
        WorkUnit("job1", "mid1", 100, 100),
        WorkUnit("job1", "mid1", 200, 100),
    ]
    
    batch = Batch.from_work_units("cpu_0", units)
    assert batch.device_id == "cpu_0"
    assert batch.job_id == "job1"
    assert batch.midstate_id == "mid1"
    assert batch.total_nonce_count == 300
    
    # Test mismatched job_ids
    bad_units = [
        WorkUnit("job1", "mid1", 0, 100),
        WorkUnit("job2", "mid1", 100, 100),
    ]
    try:
        Batch.from_work_units("cpu_0", bad_units)
        assert False, "Should have raised ValueError for mismatched job_ids"
    except ValueError:
        pass
    
    print("✅ Batch tests passed")


def test_device():
    print("Testing Device...")
    device = Device(id="test_cpu", type=DeviceType.CPU, perf_score=10.0)
    
    # Test batch size adjustment
    original_size = device.batch_size
    device.last_latency = 2.0  # Slow
    device.adjust_batch_size(1.0)  # Target 1.0s
    assert device.batch_size < original_size, "Batch size should decrease for slow execution"
    
    device.last_latency = 0.5  # Fast
    device.adjust_batch_size(1.0)
    # Should increase (though might hit min/max limits)
    
    print("✅ Device tests passed")


def test_device_manager():
    print("Testing DeviceManager...")
    dm = DeviceManager()
    
    devices = dm.get_all_devices()
    assert len(devices) > 0, "Should detect at least one device (CPU)"
    
    cpu_devices = dm.get_devices_by_type(DeviceType.CPU)
    assert len(cpu_devices) > 0, "Should detect at least one CPU"
    
    print(f"✅ DeviceManager tests passed (detected {len(devices)} device(s))")


def test_precompute_cache():
    print("Testing PrecomputeCache...")
    cache = PrecomputeCache()
    
    # Test constants
    assert 'sha256_k' in cache.constants
    assert len(cache.constants['sha256_k']) == 64
    
    # Test midstate computation
    header = secrets.token_bytes(76)
    mid_id = cache.compute_midstate("job1", header)
    assert mid_id in cache.midstates
    
    # Test retrieval
    payload = cache.get_midstate_payload(mid_id)
    assert payload is not None
    assert 'header_prefix' in payload
    assert payload['header_prefix'] == header
    
    # Test cache clearing
    cache.compute_midstate("job2", secrets.token_bytes(76))
    assert len(cache.midstates) == 2
    cache.clear_old_midstates("job1")
    assert len(cache.midstates) == 1
    
    print("✅ PrecomputeCache tests passed")


def test_hash_core():
    print("Testing HashCore...")
    hash_core = HashCore(max_workers=2)
    
    # Create test batch
    unit = WorkUnit("test_job", "mid1", 0, 1000)
    batch = Batch.from_work_units("cpu_0", [unit])
    
    # Create midstate payload
    header_prefix = secrets.token_bytes(76)
    midstate_payload = {
        'header_prefix': header_prefix,
        'partial_state': secrets.token_bytes(32),
    }
    
    # Very low difficulty so we definitely find shares
    target = (1 << (256 - 8)) - 1
    
    # Submit batch
    future = hash_core.submit_batch(batch, midstate_payload, target, DeviceType.CPU)
    result = future.result(timeout=10)
    
    assert 'device_id' in result
    assert 'hashes_processed' in result
    assert result['hashes_processed'] == 1000
    assert 'elapsed' in result
    assert result['elapsed'] > 0
    
    hash_core.shutdown()
    print("✅ HashCore tests passed")


def test_protocol_layer():
    print("Testing ProtocolLayer...")
    protocol = ProtocolLayer()
    
    # Test job submission
    job = MiningJob(
        job_id="test_job",
        header_prefix=secrets.token_bytes(76),
        target=(1 << 240),
        nonce_start=0,
        nonce_end=1000000,
    )
    
    job_received = False
    def callback(j):
        nonlocal job_received
        job_received = True
        assert j.job_id == "test_job"
    
    protocol.register_job_callback(callback)
    protocol.submit_job(job)
    
    time.sleep(0.1)  # Give callback time to fire
    assert job_received, "Job callback should have been called"
    
    # Test share submission
    share = {'nonce': 12345, 'hash': 'abcd', 'difficulty': 1.5}
    protocol.simulate_pool_submission(share)
    
    retrieved_share = protocol.wait_for_share(timeout=1.0)
    assert retrieved_share is not None
    assert retrieved_share['nonce'] == 12345
    
    protocol.shutdown()
    print("✅ ProtocolLayer tests passed")


def test_telemetry():
    print("Testing TelemetryController...")
    telemetry = TelemetryController(window_size=10)
    
    # Record some results
    for i in range(5):
        result = {
            'device_id': 'cpu_0',
            'hashes_processed': 10000,
            'elapsed': 0.1,
            'shares_found': [],
            'error_count': 0,
        }
        telemetry.record_result(result)
    
    # Check aggregated stats
    hashrate = telemetry.get_device_avg_hashrate('cpu_0')
    assert hashrate > 0, "Should have positive hashrate"
    
    latency = telemetry.get_device_avg_latency('cpu_0')
    assert abs(latency - 0.1) < 0.01, f"Expected latency ~0.1, got {latency}"
    
    summary = telemetry.get_summary()
    assert summary['total_hashes'] == 50000
    
    print("✅ TelemetryController tests passed")


def test_full_integration():
    print("Testing full integration...")
    
    # Initialize all components
    dm = DeviceManager()
    protocol = ProtocolLayer()
    hash_core = HashCore(max_workers=2)
    cache = PrecomputeCache()
    telemetry = TelemetryController(window_size=10)
    devices = dm.get_all_devices()[:1]  # Just use one device for test
    
    scheduler = SyntheticGPUScheduler(
        protocol=protocol,
        hash_core=hash_core,
        precompute=cache,
        telemetry=telemetry,
        devices=devices,
    )
    
    # Submit a job
    job = MiningJob(
        job_id="integration_test",
        header_prefix=secrets.token_bytes(76),
        target=(1 << (256 - 8)) - 1,  # Very low difficulty
        nonce_start=0,
        nonce_end=10000,
    )
    protocol.submit_job(job)
    
    # Let it run for a bit
    time.sleep(2)
    
    # Check for shares
    shares_found = 0
    deadline = time.time() + 5
    while time.time() < deadline:
        share = protocol.wait_for_share(timeout=0.5)
        if share:
            shares_found += 1
            assert 'nonce' in share
            assert 'device_id' in share
        if shares_found >= 3:  # Found enough shares
            break
    
    assert shares_found > 0, "Should have found at least one share"
    
    # Check telemetry
    summary = telemetry.get_summary()
    assert summary['total_hashes'] > 0, "Should have processed some hashes"
    assert summary['global_hashrate'] > 0, "Should have positive hashrate"
    
    scheduler.shutdown()
    print(f"✅ Full integration test passed ({shares_found} shares found, "
          f"{summary['total_hashes']:,} hashes, {summary['global_hashrate']/1e6:.2f} MH/s)")


def run_all_tests():
    print("\n" + "="*80)
    print("🧪 SYNTHETIC GPU MINER - Test Suite")
    print("="*80 + "\n")
    
    tests = [
        test_work_unit,
        test_batch,
        test_device,
        test_device_manager,
        test_precompute_cache,
        test_hash_core,
        test_protocol_layer,
        test_telemetry,
        test_full_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*80)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
