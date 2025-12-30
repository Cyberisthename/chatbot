# Negative Information Experiment

## Constraint-Based Quantum State Tracking

### Overview

This experiment implements a novel approach to quantum state tracking by focusing on **where a particle is NOT** rather than where it IS. Instead of performing direct measurements that collapse the wavefunction instantaneously, we apply **negative constraints** that progressively narrow the state space while preserving coherence.

### Motivation

Traditional quantum measurement causes instantaneous wavefunction collapse. This experiment explores whether:
1. Information can be gained without direct observation
2. Collapse can be treated as a gradual process rather than a single event
3. There exists a fundamental bound on information accessible through negative constraints

### Integration with JARVIS Architecture

This experiment uses the existing SyntheticQuantumEngine modular architecture:

- **State Evolution**: Deterministic unitary evolution with configurable dynamics
- **Artifact Logging**: Every exclusion event is logged with timestamps, state hashes, and metrics
- **Replay Capability**: Full reproducibility via seed-based determinism
- **Branching**: Three parallel execution paths for controlled comparison
- **Modular Adapters**: Automatic adapter creation linked to experiment artifacts

### Experimental Procedure

#### 1. Initialization
```python
state = |0...0⟩  # Single-particle quantum state
dim = 2^n_qubits  # State space dimension
```

#### 2. Three Branches

**Branch A: Baseline**
- Pure state evolution with no intervention
- No measurements or exclusions
- Serves as control for natural dispersion

**Branch B: Exclusion-Only**
- Apply negative constraints at regular intervals
- Identify low-probability regions and attenuate their amplitudes
- Renormalize state after each exclusion
- Track cumulative information gain

**Branch C: Direct Measurement**
- Perform projective position measurements at matched intervals
- Collapse state to eigenstate
- Compare with exclusion branch

#### 3. Negative Constraint Application

At each exclusion event:
```python
1. Compute probability distribution: p_i = |ψ_i|²
2. Select low-probability region: R = {i : p_i < threshold}
3. Attenuate amplitudes: ψ'_i = ψ_i × (1 - strength) for i ∈ R
4. Renormalize: ψ'' = ψ' / ||ψ'||
5. Log event with metadata
```

#### 4. Metrics Computed

- **Entropy**: Von Neumann entropy S = -Σ p_i log₂(p_i)
- **Support Size**: Number of basis states with significant amplitude
- **Information Gain**: Entropy reduction relative to baseline
- **Divergence**: Cumulative difference between branch trajectories
- **Saturation Point**: Step where entropy stops decreasing

### Output

Each experiment generates a `QuantumArtifact` containing:

```json
{
  "artifact_id": "neg_info_XXXXXXXX",
  "experiment_type": "negative_information_experiment",
  "results": {
    "branch_a_baseline": {
      "trajectory": [...],
      "entropy_history": [...],
      "events": []
    },
    "branch_b_exclusion": {
      "trajectory": [...],
      "entropy_history": [...],
      "events": [
        {
          "timestamp": 5,
          "excluded_region": {
            "indices": [...],
            "fraction_excluded": 0.3
          },
          "state_hash_after": "...",
          "entropy_after": 2.134
        }
      ]
    },
    "branch_c_measurement": {
      "trajectory": [...],
      "entropy_history": [...],
      "events": [...]
    },
    "comparative_metrics": {
      "info_gain_exclusion": 1.234,
      "info_gain_measurement": 3.456,
      "exclusion_vs_measurement_ratio": 0.357,
      "saturation_point": 15,
      "divergence_exclusion_measurement": 12.34
    },
    "analysis": "..."
  }
}
```

### Analysis Framework

The experiment generates an interpretation-free analysis:

**What changed:**
- Quantify entropy reduction from exclusion vs measurement
- Compare information gain between approaches

**What could not change:**
- Identify saturation point (irreducible uncertainty)
- Measure final support size differences

**What information is fundamentally inaccessible:**
- Document divergence between exclusion and measurement branches
- Observe coherence preservation vs destruction

### Usage

#### Via Script
```bash
python scripts/run_negative_info_experiment.py
```

#### Via API
```bash
curl -X POST http://localhost:3001/quantum/experiment \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_type": "negative_information_experiment",
    "config": {
      "seed": 42,
      "parameters": {
        "n_qubits": 4,
        "n_steps": 30,
        "exclusion_interval": 5,
        "exclusion_strength": 0.8,
        "evolution_type": "random_walk"
      }
    }
  }'
```

#### Via Python API
```python
from src.quantum.synthetic_quantum import SyntheticQuantumEngine, ExperimentConfig
from src.core.adapter_engine import AdapterEngine

# Initialize engines
adapter_engine = AdapterEngine(config)
quantum_engine = SyntheticQuantumEngine("./artifacts", adapter_engine)

# Configure experiment
config = ExperimentConfig(
    experiment_type="negative_information_experiment",
    seed=42,
    parameters={
        "n_qubits": 4,
        "n_steps": 30,
        "exclusion_interval": 5,
        "exclusion_strength": 0.8
    }
)

# Run experiment
artifact = quantum_engine.run_negative_information_experiment(config)

# Access results
metrics = artifact.results["comparative_metrics"]
print(f"Information gain: {metrics['info_gain_exclusion']} bits")
```

### Replay and Verification

All experiments are fully reproducible:

```python
# Load artifact by ID
artifact = quantum_engine.replay_artifact("neg_info_abc12345")

# Verify determinism by re-running with same seed
config2 = ExperimentConfig(
    experiment_type="negative_information_experiment",
    seed=42,  # Same seed
    parameters={...}  # Same parameters
)
artifact2 = quantum_engine.run_negative_information_experiment(config2)

# State hashes should match
assert artifact.results['branch_b_exclusion']['trajectory'][10]['state_hash'] == \
       artifact2.results['branch_b_exclusion']['trajectory'][10]['state_hash']
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_qubits` | int | 4 | Number of qubits (state dimension = 2^n) |
| `n_steps` | int | 20 | Number of evolution steps |
| `exclusion_interval` | int | 5 | Steps between exclusion events |
| `exclusion_strength` | float | 0.8 | Amplitude attenuation factor (0-1) |
| `evolution_type` | str | "random_walk" | Evolution dynamics ("random_walk", "dispersive", "free") |
| `seed` | int | None | Random seed for reproducibility |

### Scientific Observations

This experiment allows observation of:

1. **Information gain without direct measurement**: Quantify how much information can be extracted through exclusion alone
2. **Gradual vs instantaneous collapse**: Compare progressive narrowing (exclusion) with sudden collapse (measurement)
3. **Fundamental information bounds**: Identify saturation points where negative constraints cease to provide information
4. **Coherence preservation**: Measure how exclusion maintains superposition while measurement destroys it

### Constraints and Limitations

- No interpretation-specific assumptions (Copenhagen, Many-Worlds, etc.)
- Pure bookkeeping approach - let the data speak
- Does not assume collapse as a primitive
- Does not inject hidden variables or non-local effects

### Related Experiments

- **Quantum Reversal** (`quantacap/experiments/timerev/quantum_reversal.py`): Time-reversal fidelity under noise
- **Bell Pair Simulation**: Entanglement correlations
- **CHSH Test**: Inequality violations

### References

This experiment is inspired by:
- Weak measurement theory
- Quantum Zeno effect
- Protective measurement frameworks
- Negative probability approaches

### Future Extensions

Potential enhancements:
- Adaptive exclusion strategies (machine learning)
- Multi-particle systems
- Entangled state tracking
- Comparison with real quantum hardware results
