# Synthetic Quantum GPU

A modular Python stack that layers FLOP compression, quantum-inspired approximation, heterogeneous adapter scheduling, and infinite memory routing into a unified "synthetic quantum GPU" interface.

## Architecture Layers

1. **FLOP Compression Layer (`flop_compression.py`)**
   - Compresses dense linear operators using low-rank SVD.
   - Provides deterministic caching for expensive function calls.
   - Includes a `@flop_cached` decorator for memoization.

2. **Quantum Approximation Layer (`quantum_approx.py`)**
   - Emulates quantum-style branching with deterministic noise.
   - Performs interference using score-weighted softmax.
   - Collapses to the top branches while preserving phase information.

3. **Synthetic Adapter Cluster (`adapter_cluster.py`)**
   - Generalized scheduler for heterogeneous compute devices.
   - Supports CPU, GPU, and virtual devices with adaptive batching.
   - Pluggable work handlers and simulated performance modeling.

4. **Infinite Memory Router (`infinite_router.py`)**
   - Deterministic cosine-similarity router with snapshot support.
   - Maintains an effectively unbounded memory (with FIFO eviction).

5. **Whole-Stack Orchestrator (`orchestrator.py`)**
   - Wires all layers into the `SyntheticQuantumGPU` interface.
   - Handles `linear_op`, `branch_and_interfere`, and `cached_function` tasks.
   - Records task embeddings into the memory router for reuse.

## Installation

This package requires Python 3.10+ and NumPy.

```bash
pip install numpy
```

## Running Tests

The test suite uses the built-in `unittest` framework. From the repository root:

```bash
python -m unittest discover synthetic_quantum_gpu/tests
```

## Example Usage

```python
import numpy as np
from synthetic_quantum_gpu import SyntheticQuantumGPU

sqgpu = SyntheticQuantumGPU()

# Linear operation task
matrix = np.random.default_rng(42).normal(size=(4, 4))
vec = np.ones(4)
result = sqgpu.submit_task({
    "id": "linop_1",
    "kind": "linear_op",
    "matrix": matrix,
    "input": vec,
})
print("Linear op result:", result["result"])

# Branch and interfere task
variations = [
    {"score": 1.0},
    {"score": 0.5},
    {"score": 2.0},
]
branch_result = sqgpu.submit_task({
    "id": "branch_1",
    "kind": "branch_and_interfere",
    "base_payload": {"score": 1.0},
    "variations": variations,
    "top_k": 2,
})
print("Collapsed branches:", branch_result["branches"])

# Cached function task
cached_result = sqgpu.submit_task({
    "id": "cached_1",
    "kind": "cached_function",
    "cache_key": "expensive_const",
    "fn": lambda: 1234,
})
print("Cached result:", cached_result["result"])

sqgpu.shutdown()
```

---

The Synthetic Quantum GPU stack is designed for research and extensibility—use it as a foundation for experimenting with novel compute and scheduling strategies.
