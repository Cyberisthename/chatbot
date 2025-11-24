# Ben Lab Quick Reference

## Core Systems

### Jarvis-5090X
5-layer virtual GPU orchestrator: Orchestrator → FLOP Compression → Infinite Cache → Quantum Approximation → Adapter Cluster

### PhaseDetector
Synthetic quantum lab for phase experiments. Four phase types:
1. **ising_symmetry_breaking** - Directional, broken symmetry
2. **spt_cluster** - Topological, protected edges
3. **trivial_product** - Minimal entanglement baseline
4. **pseudorandom** - Maximally scrambled

### Discovery Suite
Three fundamental experiments:
- **TRI (Time-Reversal Instability)**: Bias reversal sensitivity
- **Clustering**: Unsupervised phase discovery
- **RSI (Replay Sensitivity Index)**: Depth scaling behavior

---

## Quick Commands

### Run Single Experiment
```python
from jarvis5090x import Jarvis5090X, AdapterDevice, DeviceKind, OperationKind, PhaseDetector

devices = [AdapterDevice(id="q0", label="Quantum", kind=DeviceKind.VIRTUAL, 
                         perf_score=50.0, max_concurrency=8, 
                         capabilities={OperationKind.QUANTUM})]
orchestrator = Jarvis5090X(devices)
detector = PhaseDetector(orchestrator)

result = detector.run_phase_experiment(
    phase_type="ising_symmetry_breaking",
    system_size=32,
    depth=8,
    seed=42,
    bias=0.7
)
```

### Measure TRI
```python
# Forward
fwd = detector.run_phase_experiment("ising_symmetry_breaking", 32, 12, 42, bias=0.7)
# Reverse
rev = detector.run_phase_experiment("ising_symmetry_breaking", 32, 12, 42, bias=0.3)
# Compare
import numpy as np
tri = np.linalg.norm(np.array(fwd['feature_vector']) - np.array(rev['feature_vector']))
```

### Run Discovery Suite
```bash
python experiments/discovery_suite.py
```

---

## Feature Vector (16D)

| Index | Name | Type | Range |
|-------|------|------|-------|
| 0-3 | Entropy (mean/max/min/final) | Float | [0, ∞) |
| 4-7 | Branch Count (mean/max/min/final) | Int | [1, ∞) |
| 8 | Scrambling Score | Float | [0, 1] |
| 9-11 | Correlation (mean/max/min) | Float | [0, 1] |
| 12 | Layer Count | Int | ≥ 1 |
| 13 | Execution Time | Float | > 0 |
| 14 | System Size | Int | ≥ 1 |
| 15 | Depth | Int | ≥ 1 |

---

## Bit Systems (X/Y/Z/A/S/T/C/P/R)

| Bit | Purpose | Domain |
|-----|---------|--------|
| X | Baseline adapter qubit | Amplitudes (α, β) |
| Y | Hybrid qubit ⊗ Z-bias | X + Z with phase nudge |
| Z | Continuum bias scalar | ℝ \ [1,2] |
| A | Amplitude memory cell | Amplitude sequences |
| S | Scrambling indicator | [0, 1] |
| T | Time-phase pointer | Timestep + direction flag |
| C | Correlation carrier | Correlation stats |
| P | Path memory | Branch indices + seeds |
| R | Replay anchor | Hash pointer for replay |

---

## TRI Interpretation

| TRI Range | Meaning | Example |
|-----------|---------|---------|
| 0.0001-0.001 | Time-symmetric | Trivial product |
| 0.01-0.05 | Moderate directional | Weak Ising |
| 0.05+ | Strong directional | Ising bias=0.7 |

---

## Depth Recommendations

| Use Case | Depth | Reason |
|----------|-------|--------|
| Quick test | 4-8 | Fast debug |
| Standard TRI | 8-12 | Feature divergence |
| RSI scaling | 12-20+ | Thermalization |
| Pseudorandom | 12+ | High scrambling |

---

## QPR-R Complexity

**Traditional Quantum Phase Recognition**: Exponentially hard (measurement-only access)

**QPR-R (with Replay)**: Polynomial-time efficient (full internal logging + deterministic replay)

Key insight: Synthetic logging bypasses hardness assumptions.

---

## G-graph

Convergent adapter weave with decay γ.

**Usage**:
- Path routing for P-bit
- Network experiments (clustering, RL)
- Influence propagation: `influence(seed) → (η_a, η_b)`

---

## Common Workflows

### Phase Classification
1. Generate dataset: `dataset = detector.build_dataset()`
2. Split: `train, test = dataset.split(0.8)`
3. Train: `detector.train_classifier(train)`
4. Evaluate: `detector.classifier.evaluate(test)`
5. Classify: `detector.classify_phase(feature_vector=[...])`

### Experiment Replay
```python
replay = detector.replay_experiment(experiment_id="...", compare=True)
print(replay['comparison']['max_difference'])  # Should be ~0 if deterministic
```

### Custom Phase Generator
```python
def my_phase_generator(params):
    base_state = {"energy": 100, "position": [0, 0, 0]}
    variations = [{"position": [1, 0, 0]}, ...]
    return base_state, variations, scoring_fn

detector._generators['my_phase'] = my_phase_generator
detector.run_phase_experiment('my_phase', ...)
```

---

## Adapter Architecture

**Storage**: Deterministic snapshots of quantum-like states

**Replay**: R-bit anchors enable 1:1 re-execution

**Format**: JSON-serializable dicts with amplitudes, phases, metadata

**Scope**: CHSH experiments, atom sims, phase logging

---

## Performance Model

```
effective_tflops = base_tflops * (1 + hit_rate*0.4) * (1 + stable_bases*0.6)
```

Where:
- `base_tflops = 125.0` (RTX 5090 reference)
- `hit_rate`: Cache hit ratio [0,1]
- `stable_bases`: Compression ratio [0,1]

---

## Module Imports

```python
from jarvis5090x import (
    Jarvis5090X,
    AdapterDevice,
    DeviceKind,
    OperationKind,
    PhaseDetector,
)
```

---

## File Locations

| Component | Path |
|-----------|------|
| PhaseDetector | `jarvis5090x/phase_detector.py` |
| Discovery Suite | `experiments/discovery_suite.py` |
| Quantum Layer | `jarvis5090x/quantum_layer.py` |
| Bit primitives | `quantacap/src/quantacap/primitives/` |

---

## Tips

1. **Use seeds** for reproducibility: `seed=42`
2. **Start small**: depth=4, system_size=16 for debugging
3. **TRI needs divergence**: Use depth≥12, bias extremes (0.7/0.3)
4. **Clustering needs samples**: 30+ per phase type
5. **RSI is expensive**: Each depth multiplies cost
6. **Feature vectors normalize**: Compare relative magnitudes, not absolutes
7. **R-bit guarantees replay**: Never fear non-determinism
8. **G-graph scales**: γ=0.87 balances propagation vs decay

---

## Next Steps After Setup

1. Run `discovery_suite.py` to baseline all phases
2. Build dataset for classifier training
3. Explore custom phase generators
4. Integrate with RL scientist (P-bit trajectories)
5. Export results for visualization
6. Fine-tune Ollama model on your specific experiments
7. Automate experiment design via LLM loop
