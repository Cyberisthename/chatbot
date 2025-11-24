#!/usr/bin/env python3
"""Test runner for Jarvis-5090X package"""

import sys


def run_test_module(module_name, test_functions):
    print(f"\n{'=' * 70}")
    print(f"  Testing: {module_name}")
    print('=' * 70)

    passed = 0
    failed = 0

    for test_fn in test_functions:
        test_name = test_fn.__name__
        try:
            test_fn()
            print(f"  ✓ {test_name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_name}: {exc!r}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return passed, failed


def main():
    print("=" * 70)
    print("  JARVIS-5090X TEST SUITE")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    # Test 1: Flop Compression Layer
    from jarvis5090x.tests import test_flop_compression
    p, f = run_test_module(
        "FlopCompressionLayer",
        [
            test_flop_compression.test_compression_layer_learns_and_compresses,
            test_flop_compression.test_compression_reuses_basis_for_similar_payloads,
            test_flop_compression.test_compression_stats_structure,
        ]
    )
    total_passed += p
    total_failed += f

    # Test 2: Infinite Cache
    from jarvis5090x.tests import test_infinite_cache
    p, f = run_test_module(
        "InfiniteMemoryCache",
        [
            test_infinite_cache.test_cache_store_and_lookup,
            test_infinite_cache.test_cache_eviction,
        ]
    )
    total_passed += p
    total_failed += f

    # Test 3: Quantum Layer
    from jarvis5090x.tests import test_quantum_layer
    p, f = run_test_module(
        "QuantumApproximationLayer",
        [
            test_quantum_layer.test_quantum_spawn,
            test_quantum_layer.test_quantum_interfere,
            test_quantum_layer.test_quantum_collapse,
            test_quantum_layer.test_quantum_deterministic,
        ]
    )
    total_passed += p
    total_failed += f

    # Test 4: Orchestrator
    from jarvis5090x.tests import test_orchestrator
    p, f = run_test_module(
        "Jarvis5090X Orchestrator",
        [
            test_orchestrator.test_orchestrator_cache_hits,
            test_orchestrator.test_orchestrator_quantum_job,
        ]
    )
    total_passed += p
    total_failed += f

    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
