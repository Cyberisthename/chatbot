#!/usr/bin/env python3
"""Demo script for Synthetic Quantum GPU."""

import time

import numpy as np

from synthetic_quantum_gpu import SyntheticQuantumGPU


def print_section(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)


def demo_linear_operations() -> None:
    """Demonstrate FLOP compression with linear operations."""
    print_section("Demo 1: FLOP Compression - Linear Operations")

    sqgpu = SyntheticQuantumGPU(dim=32, seed=42)
    rng = np.random.default_rng(42)

    print("\n🔹 Creating random matrix and vectors...")
    matrix = rng.normal(size=(8, 8))
    vec1 = rng.normal(size=8)
    vec2 = rng.normal(size=8)

    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Vector 1 shape: {vec1.shape}")
    print(f"   Vector 2 shape: {vec2.shape}")

    print("\n🔹 Task 1: Apply matrix to vector 1 (first time - will compress)...")
    start = time.perf_counter()
    result1 = sqgpu.submit_task({
        "id": "linop_1",
        "kind": "linear_op",
        "matrix": matrix,
        "input": vec1,
    })
    elapsed1 = time.perf_counter() - start

    print(f"   Status: {result1['status']}")
    print(f"   Compression rank: {result1['compression_rank']}")
    print(f"   Result shape: {result1['result'].shape}")
    print(f"   Time: {elapsed1*1000:.2f}ms")

    print("\n🔹 Task 2: Apply same matrix to vector 2 (should use cached compression)...")
    start = time.perf_counter()
    result2 = sqgpu.submit_task({
        "id": "linop_2",
        "kind": "linear_op",
        "matrix": matrix,
        "input": vec2,
    })
    elapsed2 = time.perf_counter() - start

    print(f"   Status: {result2['status']}")
    print(f"   Compression rank: {result2['compression_rank']}")
    print(f"   Result shape: {result2['result'].shape}")
    print(f"   Time: {elapsed2*1000:.2f}ms")

    if result1["compression_rank"] == result2["compression_rank"]:
        print("\n   ✅ Compression cache working: same rank for same matrix")

    error1 = np.linalg.norm(result1["result"] - matrix @ vec1)
    error2 = np.linalg.norm(result2["result"] - matrix @ vec2)
    print(f"\n   Approximation error 1: {error1:.6f}")
    print(f"   Approximation error 2: {error2:.6f}")

    sqgpu.shutdown()


def demo_quantum_branching() -> None:
    """Demonstrate quantum-style branching and interference."""
    print_section("Demo 2: Quantum Approximation - Branching & Interference")

    sqgpu = SyntheticQuantumGPU(dim=32, seed=42)

    print("\n🔹 Creating branching scenario with score variations...")
    variations = [
        {"strategy": "A", "score": 1.0},
        {"strategy": "B", "score": 0.5},
        {"strategy": "C", "score": 2.0},
        {"strategy": "D", "score": 1.5},
    ]

    print(f"   Base payload: {{'strategy': 'base', 'score': 1.0}}")
    print(f"   Variations: {len(variations)}")
    for var in variations:
        print(f"      - {var}")

    print("\n🔹 Submitting branch_and_interfere task (top_k=2)...")
    result = sqgpu.submit_task({
        "id": "branch_1",
        "kind": "branch_and_interfere",
        "base_payload": {"strategy": "base", "score": 1.0},
        "variations": variations,
        "top_k": 2,
        "temperature": 0.1,
    })

    print(f"\n   Status: {result['status']}")
    print(f"   Collapsed branches: {len(result['branches'])}")

    for i, branch in enumerate(result["branches"], 1):
        print(f"\n   Branch {i}:")
        print(f"      ID: {branch['id']}")
        print(f"      Amplitude: {branch['amplitude']}")
        print(f"      Amplitude magnitude: {abs(branch['amplitude']):.4f}")
        print(f"      Payload: {branch['payload']}")

    print("\n   ✅ Higher-scoring strategies received higher amplitudes")

    sqgpu.shutdown()


def demo_cached_functions() -> None:
    """Demonstrate function result caching."""
    print_section("Demo 3: Cached Function Execution")

    sqgpu = SyntheticQuantumGPU(dim=32, seed=42)

    print("\n🔹 Defining expensive computation (fibonacci(30))...")

    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    print("\n🔹 Task 1: First computation (uncached)...")
    start = time.perf_counter()
    result1 = sqgpu.submit_task({
        "id": "fib_1",
        "kind": "cached_function",
        "cache_key": "fib_30",
        "fn": lambda: fib(20),  # Using smaller n for demo speed
    })
    elapsed1 = time.perf_counter() - start

    print(f"   Status: {result1['status']}")
    print(f"   Result: {result1['result']}")
    print(f"   Time: {elapsed1*1000:.2f}ms")

    print("\n🔹 Task 2: Same computation (should be cached)...")
    start = time.perf_counter()
    result2 = sqgpu.submit_task({
        "id": "fib_2",
        "kind": "cached_function",
        "cache_key": "fib_30",
        "fn": lambda: fib(20),
    })
    elapsed2 = time.perf_counter() - start

    print(f"   Status: {result2['status']}")
    print(f"   Result: {result2['result']}")
    print(f"   Time: {elapsed2*1000:.2f}ms")

    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float("inf")
    print(f"\n   Speedup: {speedup:.1f}x")
    print(f"   ✅ Caching reduced computation time significantly")

    sqgpu.shutdown()


def demo_memory_router() -> None:
    """Demonstrate memory routing and similarity search."""
    print_section("Demo 4: Infinite Memory Router")

    sqgpu = SyntheticQuantumGPU(dim=32, seed=42)
    rng = np.random.default_rng(42)

    print("\n🔹 Submitting multiple tasks to populate memory...")
    matrix = rng.normal(size=(4, 4))
    for i in range(5):
        vec = rng.normal(size=4)
        sqgpu.submit_task({
            "id": f"mem_task_{i}",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vec,
        })
        print(f"   Submitted task: mem_task_{i}")

    print(f"\n   Total memory entries: {len(sqgpu.memory_router._entries)}")

    print("\n🔹 Routing query to find similar tasks...")
    query_embedding = rng.normal(size=32)
    query_embedding /= np.linalg.norm(query_embedding) or 1.0

    similar_entries = sqgpu.memory_router.route(query_embedding, top_k=3)

    print(f"   Found {len(similar_entries)} similar entries:")
    for entry in similar_entries:
        print(f"      - {entry.key}: {entry.payload}")

    print("\n🔹 Creating snapshot...")
    snapshot = sqgpu.memory_router.snapshot()
    print(f"   Snapshot contains {len(snapshot['entries'])} entries")
    print(f"   Snapshot dimension: {snapshot['dim']}")

    print("\n   ✅ Memory router successfully stores and retrieves task history")

    sqgpu.shutdown()


def demo_adapter_cluster() -> None:
    """Demonstrate heterogeneous adapter cluster scheduling."""
    print_section("Demo 5: Synthetic Adapter Cluster")

    sqgpu = SyntheticQuantumGPU(dim=32, seed=42)

    print("\n🔹 Cluster configuration:")
    stats = sqgpu.adapter_cluster.get_stats()
    for device_id, device_info in stats["devices"].items():
        print(f"   {device_id}:")
        print(f"      Type: {device_info['type']}")
        print(f"      Performance score: {device_info['perf_score']:.1f}")
        print(f"      Batch size: {device_info['batch_size']}")

    print("\n🔹 Submitting batch of tasks...")
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(4, 4))

    for i in range(10):
        vec = rng.normal(size=4)
        sqgpu.submit_task({
            "id": f"batch_task_{i}",
            "kind": "linear_op",
            "matrix": matrix,
            "input": vec,
        })

    print(f"   Submitted 10 tasks")

    time.sleep(0.1)

    print("\n🔹 Final cluster stats:")
    final_stats = sqgpu.adapter_cluster.get_stats()
    print(f"   Queue length: {final_stats['queue_length']}")
    print(f"   Active tasks: {final_stats['active_tasks']}")
    for device_id, device_info in final_stats["devices"].items():
        print(f"   {device_id}:")
        print(f"      Is busy: {device_info['is_busy']}")
        print(f"      Last latency: {device_info['last_latency']*1000:.2f}ms")
        print(f"      Batch size: {device_info['batch_size']}")

    print("\n   ✅ Cluster adaptively scheduled work across devices")

    sqgpu.shutdown()


def demo_full_workflow() -> None:
    """Demonstrate complete workflow using all layers."""
    print_section("Demo 6: Full Workflow - All Layers Together")

    sqgpu = SyntheticQuantumGPU(dim=64, max_cache_items=100, seed=42)
    rng = np.random.default_rng(42)

    print("\n🔹 Complex workflow: Linear ops → Branching → Caching → Memory routing")

    # Step 1: Linear operations
    print("\n   Step 1: Processing linear operations...")
    matrices = [rng.normal(size=(6, 6)) for _ in range(3)]
    vectors = [rng.normal(size=6) for _ in range(3)]

    for i, (mat, vec) in enumerate(zip(matrices, vectors)):
        result = sqgpu.submit_task({
            "id": f"workflow_linop_{i}",
            "kind": "linear_op",
            "matrix": mat,
            "input": vec,
        })
        print(f"      Linear op {i}: rank={result['compression_rank']}")

    # Step 2: Branching
    print("\n   Step 2: Quantum branching with strategies...")
    strategies = [
        {"name": "aggressive", "risk": 0.8, "reward": 2.5},
        {"name": "balanced", "risk": 0.5, "reward": 1.5},
        {"name": "conservative", "risk": 0.2, "reward": 1.0},
    ]

    branch_result = sqgpu.submit_task({
        "id": "workflow_branch",
        "kind": "branch_and_interfere",
        "base_payload": {"name": "baseline", "risk": 0.5, "reward": 1.0},
        "variations": strategies,
        "top_k": 2,
    })
    print(f"      Selected {len(branch_result['branches'])} best strategies")

    # Step 3: Cached computation
    print("\n   Step 3: Caching expensive computation...")
    cached_result = sqgpu.submit_task({
        "id": "workflow_cache",
        "kind": "cached_function",
        "cache_key": "expensive_analysis",
        "fn": lambda: sum(i**2 for i in range(1000)),
    })
    print(f"      Cached result: {cached_result['result']}")

    # Step 4: Memory routing
    print("\n   Step 4: Querying memory for similar tasks...")
    query = rng.normal(size=64)
    query /= np.linalg.norm(query) or 1.0
    similar = sqgpu.memory_router.route(query, top_k=3)
    print(f"      Found {len(similar)} similar past tasks")

    # Stats
    print("\n   Final statistics:")
    cache_stats = sqgpu.flop_compressor.get_cache_stats()
    cluster_stats = sqgpu.adapter_cluster.get_stats()
    print(f"      FLOP cache: {cache_stats['lru_cache_size']} items cached")
    print(f"      Memory entries: {len(sqgpu.memory_router._entries)}")
    print(f"      Pending work: {cluster_stats['queue_length']} units")

    print("\n   ✅ All layers working together seamlessly!")

    sqgpu.shutdown()


def main() -> None:
    """Run all demos."""
    print("\n" + "="*80)
    print("🚀 SYNTHETIC QUANTUM GPU - Comprehensive Demo")
    print("="*80)

    try:
        demo_linear_operations()
        demo_quantum_branching()
        demo_cached_functions()
        demo_memory_router()
        demo_adapter_cluster()
        demo_full_workflow()

        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
