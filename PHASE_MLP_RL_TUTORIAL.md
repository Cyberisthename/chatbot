# Phase MLP & RL Scientist - Practical Tutorial

This tutorial walks you through using the Phase MLP classifier and RL Scientist from start to finish.

## Prerequisites

The core functionality works out-of-the-box. For neural network features:
```bash
pip install torch  # Optional: enables MLPPhaseClassifier
```

## Part 1: Generate Training Data

First, generate a dataset of phase experiments:

```bash
# Quick test (40 samples, ~30 seconds)
python experiments/build_phase_dataset.py --num-per-phase 10 --output data/test_dataset.json

# Small dataset (600 samples, ~5 minutes)
python experiments/build_phase_dataset.py --num-per-phase 150

# Large dataset (2000 samples, ~15 minutes)
python experiments/build_phase_dataset.py --num-per-phase 500 --output data/large_dataset.json
```

This generates experiments across 4 phases:
- `ising_symmetry_breaking`: Ferromagnetic-like ordered phase
- `spt_cluster`: Symmetry-protected topological phase
- `trivial_product`: Unordered product state
- `pseudorandom`: High-entropy random phase

## Part 2: Compare Classifiers

Once you have a dataset, compare different classifiers:

```bash
python experiments/evaluate_phase_classifiers.py --dataset data/phase_dataset.json
```

**Expected Output:**
```
=== Comparison Summary ===
Classifier                Accuracy     Mean Confidence    Correct/Total
================================================================================
Simple k-NN (k=5)        0.8542       0.6234             82/96
Centroid Classifier      0.8750       0.7123             84/96
MLP Neural Net           0.9583       0.8921             92/96  # If PyTorch available
```

The MLP typically achieves 5-15% higher accuracy than simpler methods on complex phase boundaries.

## Part 3: Train RL Agent

The RL agent learns which experiment settings produce high Time-Reversal Instability (TRI):

```bash
# Quick test (100 episodes)
python experiments/rl_scientist.py --episodes 100

# Better results (500+ episodes)
python experiments/rl_scientist.py --episodes 500 --seed 42
```

**What's happening:**
- Agent tries different (phase, bias, depth) combinations
- Learns which settings produce high TRI scores
- Converges on optimal experiment configurations

**Example results:**
```
Best action discovered:
  phase=ising_symmetry_breaking
  bias=0.7
  depth=12
  expected_TRI=3.8921

Top 5 actions by Q-value:
  1. ising_symmetry_breaking bias=0.7 depth=12 Q=3.8921
  2. spt_cluster             bias=0.8 depth=12 Q=2.4567
  3. ising_symmetry_breaking bias=0.6 depth=12 Q=2.1234
```

**Interpretation:** The agent discovered that `ising_symmetry_breaking` with high depth and mid-range bias produces the most "interesting" (high TRI) experiments.

## Part 4: Use in Code

### 4.1 Basic Phase Classification

```python
from jarvis5090x import (
    AdapterDevice, DeviceKind, Jarvis5090X, 
    OperationKind, PhaseDetector
)

# Setup
devices = [AdapterDevice(
    id="quantum_0", label="Simulator", 
    kind=DeviceKind.VIRTUAL, perf_score=50.0,
    max_concurrency=8, capabilities={OperationKind.QUANTUM}
)]
orchestrator = Jarvis5090X(devices)
detector = PhaseDetector(orchestrator)

# Run experiment
result = detector.run_phase_experiment(
    phase_type="ising_symmetry_breaking",
    system_size=32, depth=8, seed=42, bias=0.7
)

# Classify
dataset = detector.build_dataset()
detector.train_classifier(dataset)
classification = detector.classify_phase(
    experiment_id=result["experiment_id"]
)
print(f"Predicted: {classification['prediction']}")
print(f"Confidence: {classification['confidence']:.4f}")
```

### 4.2 Using MLP Classifier (with PyTorch)

```python
from jarvis5090x import PhaseDetector, MLPPhaseClassifier

detector = PhaseDetector(orchestrator)

# Generate some data
for phase in ["ising_symmetry_breaking", "spt_cluster"]:
    for _ in range(20):
        detector.run_phase_experiment(
            phase_type=phase, system_size=32, 
            depth=8, seed=random.randint(1, 10000)
        )

dataset = detector.build_dataset()
train_data, test_data = dataset.split(0.8)

# Train MLP
mlp = MLPPhaseClassifier()
if mlp.is_available():
    mlp.train(train_data, epochs=30, lr=1e-3)
    results = mlp.evaluate(test_data)
    print(f"MLP Accuracy: {results['accuracy']:.4f}")
else:
    print("PyTorch not available")
```

### 4.3 Custom RL Experiment

```python
from experiments.rl_scientist import RLLabEnv, train_q_agent

detector = make_detector()
env = RLLabEnv(detector)

# Custom reward function (example: prefer low TRI)
def custom_step(action_idx):
    state, reward, result = env.step(action_idx)
    custom_reward = -reward  # Invert for low TRI
    return state, custom_reward, result

# Train with custom logic
# ... (implement custom training loop)
```

## Part 5: Advanced Usage

### 5.1 Feature Engineering

Extract and inspect features:

```python
from jarvis5090x import extract_features

result = detector.run_phase_experiment(
    phase_type="spt_cluster", system_size=32, 
    depth=8, seed=123
)

experiment_id = result["experiment_id"]
features = detector.log_phase_features(experiment_id)

print(f"Feature vector ({len(features)} dims):")
for i, val in enumerate(features):
    print(f"  f{i}: {val:.4f}")
```

### 5.2 Batch Classification

```python
# Load pre-trained dataset
from jarvis5090x import PhaseDataset

dataset = PhaseDataset.load_json("data/phase_dataset.json")
train, test = dataset.split(0.8)

# Train classifier
from jarvis5090x import CentroidPhaseClassifier
classifier = CentroidPhaseClassifier()
classifier.train(train)

# Batch predict
predictions = []
for example in test.examples:
    label, conf = classifier.predict(example.feature_vector)
    predictions.append({
        "true": example.phase_label,
        "predicted": label,
        "confidence": conf
    })

# Analyze errors
errors = [p for p in predictions if p['true'] != p['predicted']]
print(f"Error rate: {len(errors) / len(predictions):.2%}")
```

### 5.3 Save/Load Datasets

```python
from jarvis5090x import PhaseDataset

# Save
dataset = detector.build_dataset()
dataset.save_json("my_experiments.json")

# Load later
loaded = PhaseDataset.load_json("my_experiments.json")
print(f"Loaded {len(loaded)} examples")

# Merge datasets
from jarvis5090x import merge_datasets
dataset1 = PhaseDataset.load_json("dataset1.json")
dataset2 = PhaseDataset.load_json("dataset2.json")
combined = merge_datasets(dataset1, dataset2)
combined.save_json("combined.json")
```

## Troubleshooting

### PyTorch Not Available

If you see warnings about PyTorch:
```bash
pip install torch
```

Or continue using centroid/k-NN classifiers (no PyTorch needed).

### Low Classification Accuracy

1. **Generate more data**: `--num-per-phase 200` or higher
2. **Use MLP**: Neural nets work better with larger datasets
3. **Check phase separation**: Some phases may be inherently similar

### RL Agent Not Learning

1. **Increase episodes**: `--episodes 1000` or more
2. **Check reward signal**: Ensure TRI values vary across actions
3. **Adjust hyperparameters**: Modify α (learning rate) or ε (exploration) in `rl_scientist.py`

## Next Steps

1. **Experiment with architectures**: Modify `PhaseMLP` to add layers or change activation functions
2. **Implement DQN**: Upgrade Q-table to neural function approximation
3. **Multi-objective RL**: Optimize for TRI + computational cost
4. **Domain LLM**: Fine-tune a small language model on your lab docs (see README)

## Examples in Action

Run the complete demo:
```bash
python demo_phase_mlp_rl.py
```

This demonstrates:
- Data generation
- Classifier comparison (Centroid vs MLP)
- RL agent training and convergence
- Best experiment discovery

Typical runtime: 2-3 minutes
