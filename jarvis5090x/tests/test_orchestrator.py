from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    FlopCompressionLayer,
    InfiniteMemoryCache,
    Jarvis5090X,
    QuantumApproximationLayer,
)
from jarvis5090x.types import OperationKind, OperationRequest


def create_orchestrator():
    devices = [
        AdapterDevice(
            id="cpu_0",
            label="HighPerf CPU",
            kind=DeviceKind.CPU,
            perf_score=10.0,
            capabilities={OperationKind.HASHING, OperationKind.LINALG, OperationKind.GENERIC},
        ),
        AdapterDevice(
            id="quantum_0",
            label="Quantum Simulator",
            kind=DeviceKind.VIRTUAL,
            perf_score=5.0,
            capabilities={OperationKind.QUANTUM},
        ),
    ]
    cluster = AdapterCluster(devices)
    compression = FlopCompressionLayer(max_bases=20, stability_threshold=2, tolerance=0.1)
    cache = InfiniteMemoryCache(max_items=100)
    quantum = QuantumApproximationLayer(max_branches=8)
    orchestrator = Jarvis5090X(
        devices=devices,
        compression_layer=compression,
        cache_layer=cache,
        quantum_layer=quantum,
        adapter_cluster=cluster,
    )
    return orchestrator, cache


def test_orchestrator_cache_hits():
    orchestrator, cache = create_orchestrator()
    payload = {
        "matrix": [[1.0, 0.0], [2.0, 3.0]],
        "vector": [5.0, 7.0],
        "operation": "matmul",
    }

    result1 = orchestrator.submit("linalg", "test_linalg", payload)
    result2 = orchestrator.submit("linalg", "test_linalg", payload)

    assert result1 == result2
    assert cache.stats()["hits"] >= 1


def test_orchestrator_quantum_job():
    orchestrator, _ = create_orchestrator()
    payload = {
        "base_state": {"energy": 10},
        "variations": [{"energy": 12}, {"energy": 8}],
        "scoring_fn": lambda state: state.get("energy", 0),
        "top_k": 1,
    }
    result = orchestrator.submit("quantum", "quantum_job", payload)
    assert "collapsed_state" in result
