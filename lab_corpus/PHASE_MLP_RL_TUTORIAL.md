# Phase MLP + RL Scientist Tutorial

## Overview

This guide explains how to turn the PhaseDetector feature vectors into:
1. **A lightweight Phase MLP classifier** for fast phase recognition
2. **An RL "lab scientist" loop** that automatically proposes new experiments based on results

The workflow is fully deterministic—every experiment, dataset build, and classifier training run is reproducible thanks to adapter replay (R-bit anchors).

---

## Phase Feature Vector Recap

Each experiment produces a 16D feature vector capturing entropy, branch counts, correlation, and system parameters.

```
[entropy_mean, entropy_max, entropy_min, entropy_final,
 branch_count_mean, branch_count_max, branch_count_min, branch_count_final,
 scrambling_score,
 correlation_mean, correlation_max, correlation_min,
 layer_count, execution_time, system_size, depth]
```

These features are logged inside `PhaseDetector.run_phase_experiment` and exported through `phase_logger`.

---

## Step 1: Build a Dataset

```python
from jarvis5090x import PhaseDetector, Jarvis5090X, AdapterDevice, DeviceKind, OperationKind

# Configure orchestrator
devices = [AdapterDevice(
    id="quantum_0",
    label="Quantum",
    kind=DeviceKind.VIRTUAL,
    perf_score=50.0,
    max_concurrency=8,
    capabilities={OperationKind.QUANTUM},
)]
orchestrator = Jarvis5090X(devices)

detector = PhaseDetector(orchestrator)

# Run experiments (collect multiple per phase)
for phase in ["ising_symmetry_breaking", "spt_cluster", "trivial_product", "pseudorandom"]:
    for depth in (6, 8, 10, 12):
        detector.run_phase_experiment(
            phase_type=phase,
            system_size=32,
            depth=depth,
            seed=depth * 11,
            bias=0.7 if phase == "ising_symmetry_breaking" else None,
        )

# Build dataset & split
full_dataset = detector.build_dataset()
train_dataset, test_dataset = full_dataset.split(ratio=0.8)
```

The dataset entries live in `jarvis5090x/phase_dataset.py` and hold:
- `experiment_id`
- `phase_label`
- `feature_vector`
- `params`

---

## Step 2: Train a Phase MLP

You can start with the built-in classifiers (`SimplePhaseClassifier`, `CentroidPhaseClassifier`). To push accuracy higher, drop the feature vectors into a tiny MLP.

### PyTorch Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert dataset → tensors
import numpy as np

label_to_idx = {label: idx for idx, label in enumerate(sorted({ex.phase_label for ex in train_dataset.examples}))}

X_train = np.array([ex.feature_vector for ex in train_dataset.examples], dtype=np.float32)
y_train = np.array([label_to_idx[ex.phase_label] for ex in train_dataset.examples], dtype=np.int64)

X_test = np.array([ex.feature_vector for ex in test_dataset.examples], dtype=np.float32)
y_test = np.array([label_to_idx[ex.phase_label] for ex in test_dataset.examples], dtype=np.int64)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
```

```python
class PhaseMLP(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = PhaseMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

```python
for epoch in range(25):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # Monitor accuracy
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = logits.argmax(dim=1)
        acc = (preds.numpy() == y_test).mean()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")
```

**Expected**: Accuracy ≥ 0.95 with just a few epochs, because the feature vector already separates phases cleanly.

Save the model + normalization stats for live inference.

---

## Step 3: Live Classification Pipeline

```python
def classify_with_mlp(model, feature_vector):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(feature_vector, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs))
        return pred_idx, probs.numpy()
```

Add mapping from indices back to labels using `idx_to_label = {idx: label for label, idx in label_to_idx.items()}`.

---

## Step 4: RL Lab Scientist Loop

The RL scientist sits on top of the MLP and PhaseDetector. The loop:

1. **State**: Feature vector + metadata from the last experiment
2. **Action**: Choice of next experiment parameters (phase, depth, bias, system size)
3. **Reward**: Improvement in a metric (e.g., TRI, classification confidence, entropy gap)
4. **Transition**: Run new experiment → get new feature vector
5. **Policy Update**: Use RL algorithm (e.g., PPO, DQN) to adjust experiment selection

### Minimal Loop Skeleton

```python
import random

phase_space = ["ising_symmetry_breaking", "spt_cluster", "trivial_product", "pseudorandom"]

def run_policy_step(state, policy_model):
    # Example: epsilon-greedy on top of policy network outputs
    if random.random() < 0.1:
        action = random.choice(phase_space)
    else:
        action = policy_model.predict(state)
    
    # Convert action → experiment params
    params = {
        "phase_type": action,
        "system_size": 32,
        "depth": random.choice([8, 10, 12, 14]),
        "seed": random.randint(0, 10_000),
        "bias": 0.7 if action == "ising_symmetry_breaking" else None,
    }
    
    result = detector.run_phase_experiment(**{k: v for k, v in params.items() if v is not None})
    feature_vector = result["feature_vector"]
    reward = compute_reward(result)
    
    transition = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": feature_vector,
        "metadata": result,
    }
    return transition
```

### Reward Ideas

- **TRI Maximization**: Reward = TRI value if action selects Ising & biases
- **Phase Coverage**: Encourage visiting under-sampled phases (information gain)
- **Classifier Confidence**: Reward = classifier confidence on predicted label
- **Feature Divergence**: Reward increases when entropy or scrambling hits new highs

### Policy Storage

Store transitions with P-bit (path memory) and R-bit anchors so the policy can replay sequences deterministically. This lets you train RL offline from stored trajectories.

---

## Looping with Ollama Lab Assistant

Combine RL loop with the LLM from `ollama/ollama_lab_integration.py`:

1. RL policy suggests candidate parameters
2. LLM reviews and tweaks design (“push depth to 14 for stronger TRI”)
3. PhaseDetector runs experiment
4. Results go back to RL buffer + LLM for interpretation
5. Repeat until reward converges

This hybrid approach keeps hard control logic in RL while leveraging the LLM for qualitative reasoning, hypothesis generation, and explanation.

---

## Exporting Models

- Save MLP weights → `phase_mlp.pt`
- Serialize normalization stats (mean/std per feature)
- Export RL policy checkpoints per training run
- Keep metadata (bias, depth ranges) for reproducibility

Add these paths to `.gitignore` if you store them locally:
```
phase_mlp.pt
rl_policy_*.pt
replay_buffers/
```

---

## Best Practices

1. **Normalize Features**: z-score each dimension before feeding the MLP.
2. **Balance Dataset**: Ensure each phase has similar sample counts.
3. **Curriculum for RL**: Start with small depth range, expand as policy stabilizes.
4. **Deterministic Seeds**: Keep `seed` consistent when comparing policies.
5. **Logging**: Record reward curves, TRI history, classifier accuracy per iteration.
6. **Integration**: Use `OllamaLabAssistant` to interpret RL policy decisions in natural language.
7. **Evaluation**: Periodically freeze policy and run a fixed benchmark set for comparison.

---

## Next Steps

- Extend the MLP to a deeper network or try gradient boosting for interpretability.
- Integrate policy gradients (PPO) with reward shaping from TRI and RSI trends.
- Feed RL outcomes back into training data to fine-tune the Ollama model on fresh discoveries.
- Automate experiment pipelines so the RL scientist can run overnight discovery sweeps.

With this pipeline, the lab gains both **fast classification** (Phase MLP) and **autonomous exploration** (RL scientist), all grounded in deterministic phase replay.
