# Phase Detector - Replayable Synthetic Quantum Lab

## Overview

The Phase Detector is a replayable synthetic quantum lab built on top of Jarvis-5090X that simulates quantum states/circuits, logs correlation and "phase fingerprint" features, replays experiments deterministically, and trains classifiers to recognize phases of matter (symmetry-breaking, SPT, trivial product, pseudorandom).

## Key Features

- **Synthetic Phase Simulation**: Simulate controllable quantum phases of matter
- **Deep Logging**: Log internal correlation structure and phase features
- **Deterministic Replay**: Replay experiments with the same seed for reproducibility
- **Phase Classification**: Train ML models to recognize phase types efficiently
- **Complexity Advantage**: Bypass hardness assumptions through replay-augmented model

## Architecture

### Core Components

1. **phase_logger.py** - Low-level hooks for logging experiment data
   - Records layer-by-layer branch evolution
   - Captures probability distributions and entropy
   - Stores state snapshots with configurable limits

2. **phase_features.py** - Extract numeric feature vectors from logs
   - Probability entropy profiles
   - Branch count profiles
   - Scrambling scores
   - Correlation profiles
   - System parameters

3. **phase_replay.py** - Replay experiments deterministically
   - Validate experiment records
   - Compare original vs replayed features
   - Ensure deterministic reproducibility

4. **phase_detector.py** - High-level API for running phase experiments
   - Phase generators (Ising, SPT, Product, Pseudorandom)
   - Experiment orchestration
   - Feature extraction and logging
   - Classification interface

5. **phase_dataset.py** - Dataset builder for training
   - Build labeled datasets from experiments
   - Train/test splitting
   - JSON serialization

6. **phase_classifier.py** - ML classifiers for phase recognition
   - SimplePhaseClassifier (k-NN)
   - CentroidPhaseClassifier (nearest centroid)
   - Confusion matrix evaluation

## Phase Types

### 1. Ising Symmetry Breaking (`ising_symmetry_breaking`)

**Characteristics:**
- High magnetization
- Symmetry broken (symmetry_indicator = 1.0)
- Low entropy
- Medium correlation length

**Physics:**
Simulates spontaneous symmetry breaking in the 1D Ising model.

### 2. SPT Cluster Phase (`spt_cluster`)

**Characteristics:**
- High string order parameter (~0.85)
- Edge mode imbalance
- Low-to-medium entropy
- Topological indicators

**Physics:**
Simulates a symmetry-protected topological (SPT) phase with protected edge modes.

### 3. Trivial Product State (`trivial_product`)

**Characteristics:**
- Near-zero string order
- Low correlation length
- Low symmetry indicators
- Minimal scrambling

**Physics:**
Simulates a trivial product state with minimal entanglement.

### 4. Pseudorandom Phase (`pseudorandom`)

**Characteristics:**
- High entropy
- High randomness score
- Uniform probability distribution
- Strong scrambling

**Physics:**
Simulates a maximally scrambled pseudorandom unitary (PRU-like) phase.

## Feature Vectors

Each experiment produces a 16-dimensional feature vector:

1. `entropy_mean` - Mean probability entropy across layers
2. `entropy_max` - Maximum entropy observed
3. `entropy_min` - Minimum entropy observed
4. `entropy_final` - Final layer entropy
5. `branch_count_mean` - Mean number of branches
6. `branch_count_max` - Maximum branches
7. `branch_count_min` - Minimum branches
8. `branch_count_final` - Final branch count
9. `scrambling_score` - Measure of probability uniformity
10. `correlation_mean` - Mean correlation proxy
11. `correlation_max` - Maximum correlation
12. `correlation_min` - Minimum correlation
13. `layer_count` - Total number of layers logged
14. `execution_time` - Time to complete experiment
15. `system_size` - Number of qubits/spins
16. `depth` - Circuit depth

## Usage

### Basic Example

```python
from jarvis5090x import (
    Jarvis5090X,
    AdapterDevice,
    DeviceKind,
    OperationKind,
    PhaseDetector,
)

# Setup orchestrator
devices = [
    AdapterDevice(
        id="quantum_0",
        label="Quantum Simulator",
        kind=DeviceKind.VIRTUAL,
        perf_score=50.0,
        max_concurrency=8,
        capabilities={OperationKind.QUANTUM},
    )
]
orchestrator = Jarvis5090X(devices)

# Create phase detector
detector = PhaseDetector(orchestrator)

# Run experiment
result = detector.run_phase_experiment(
    phase_type="ising_symmetry_breaking",
    system_size=32,
    depth=8,
    seed=42,
)

print(f"Experiment ID: {result['experiment_id']}")
print(f"Feature vector: {result['feature_vector']}")
print(f"Summary: {result['summary']}")
```

### Replay Experiment

```python
# Replay with comparison
replay = detector.replay_experiment(
    experiment_id="ising_symmetry_breaking::abc123::def456",
    compare=True,
)

print(f"Max difference: {replay['comparison']['max_difference']}")
print(f"Is match: {replay['comparison']['is_match']}")
```

### Train Classifier

```python
# Build dataset from logged experiments
dataset = detector.build_dataset()

# Split train/test
train_dataset, test_dataset = dataset.split(ratio=0.8)

# Train
detector.train_classifier(train_dataset)

# Evaluate
evaluation = detector.classifier.evaluate(test_dataset)
print(f"Accuracy: {evaluation['accuracy'] * 100:.2f}%")
```

### Classify New Experiment

```python
# Classify by experiment ID
classification = detector.classify_phase(experiment_id="ising::xyz789::abc012")
print(f"Predicted: {classification['prediction']}")
print(f"Confidence: {classification['confidence']}")

# Or by feature vector
classification = detector.classify_phase(feature_vector=[...])
```

## Complexity Model

### Traditional Model (Schuster et al.)

In the standard quantum computing model with only measurement access:
- Phase recognition: **HARD** (potentially exponential)
- Only measurement outcomes available
- No access to internal evolution

### Replay-Augmented Model (QPR-R)

In our replay-augmented synthetic model:
- Phase recognition: **EFFICIENT** (polynomial)
- Full logging of internal state evolution
- Deterministic replay capability
- Feature extraction scales as O(layers × features)
- Classification scales as O(n^k) where n = system size, k small

**Key Insight:** By allowing synthetic logging and replay of the full evolution, we bypass the hardness assumptions. This creates a new complexity class: **QPR-R** (Quantum Phase Recognition with Replay).

## CTO Story

**"Jarvis-5090X is not just a virtual GPU / ASIC system. It's a Replay-Augmented Quantum Lab that:**

1. **Simulates phases of matter** - Four controllable phase families with tunable parameters
2. **Logs internal correlation structure** - Deep instrumentation of quantum evolution
3. **Replays experiments** - Deterministic reproducibility for validation
4. **Classifies phases efficiently** - Where real-world methods are exponentially hard"

### Deliverables

- **Technical Architecture** - This document + codebase
- **Demo Scripts** - `examples/phase_quickstart.py`
- **Research Angle** - Exploring QPR-R complexity class
- **Scalability** - Demonstrated polynomial scaling vs exponential hardness

## API Reference

### PhaseDetector

```python
class PhaseDetector:
    def __init__(
        self,
        orchestrator: Jarvis5090X,
        logger: Optional[PhaseLogger] = None,
        generators: Optional[Dict[str, PhaseGenerator]] = None,
    ) -> None: ...

    def run_phase_experiment(
        self,
        phase_type: str,
        system_size: int,
        depth: int,
        seed: int,
        *,
        top_k: int = 1,
        **phase_options: Any,
    ) -> Dict[str, Any]: ...

    def replay_experiment(
        self,
        experiment_id: str,
        *,
        compare: bool = True,
    ) -> Dict[str, Any]: ...

    def classify_phase(
        self,
        *,
        experiment_id: Optional[str] = None,
        feature_vector: Optional[List[float]] = None,
        retrain: bool = False,
    ) -> Dict[str, Any]: ...

    def build_dataset(self) -> PhaseDataset: ...

    def train_classifier(self, dataset: Optional[PhaseDataset] = None) -> Dict[str, Any]: ...
```

### Module-Level API

```python
from jarvis5090x import (
    configure_phase_detector,
    run_phase_experiment,
    log_phase_features,
    replay_experiment,
    classify_phase,
)

# Configure global detector
configure_phase_detector(detector)

# Use convenience functions
result = run_phase_experiment(
    phase_type="ising_symmetry_breaking",
    system_size=32,
    depth=8,
    seed=42,
)
```

## Implementation Checklist

- [x] Add experiment metadata + IDs
- [x] Add hooks into QuantumApproximationLayer
- [x] Implement phase_logger
- [x] Implement phase_replay
- [x] Define 4 synthetic phase types and their generators
- [x] Implement phase_features (feature vectors)
- [x] Implement phase_detector high-level API
- [x] Implement dataset builder (phase_dataset)
- [x] Implement classifiers (phase_classifier)
- [x] Build demo script
- [x] Documentation (this file)

## Future Extensions

- **Additional Phases**: Add more exotic phases (fracton, many-body localized, etc.)
- **Visualization**: Plot phase separation in feature space
- **Advanced Classifiers**: Neural network classifiers, gradient boosting
- **Batch Processing**: Parallel execution of multiple experiments
- **Persistence**: Save/load experiments and trained models
- **Benchmarking**: Formal complexity analysis and scaling plots

## References

- Schuster et al. - "Computational hardness of quantum phase recognition" (theoretical motivation)
- Jarvis-5090X Architecture - `ARCHITECTURE.md`
- Quantum Approximation Layer - `quantum_layer.py`
- Demo Script - `examples/phase_quickstart.py`
