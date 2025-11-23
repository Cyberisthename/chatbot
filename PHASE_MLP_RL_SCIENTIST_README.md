# Phase MLP & RL Scientist Lab

This document describes the AI/ML components built on top of PhaseDetector for training neural phase classifiers and RL agents.

## Overview

Three main components are implemented:

1. **MLPPhaseClassifier**: A neural network that classifies quantum phases
2. **Phase Dataset Builder**: Script to generate large training datasets
3. **RL Lab Scientist**: A Q-learning agent that learns optimal experiment settings

## A. Phase Recognition with Neural Networks

### MLPPhaseClassifier

Located in `jarvis5090x/phase_mlp_classifier.py`, this classifier uses a simple 2-layer MLP to predict phase types from feature vectors.

**Architecture:**
- Input: 16-dimensional feature vector
- Hidden layers: 64 units each with ReLU activation
- Output: Softmax over 4 phase classes

**Usage:**

```python
from jarvis5090x import MLPPhaseClassifier, PhaseDataset

# Load or build dataset
dataset = PhaseDataset.load_json("data/phase_dataset.json")
train_dataset, test_dataset = dataset.split(0.8)

# Train classifier
classifier = MLPPhaseClassifier()
classifier.train(train_dataset, epochs=30, lr=1e-3)

# Predict
label, confidence = classifier.predict(feature_vector)

# Evaluate
results = classifier.evaluate(test_dataset)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Building a Dataset

Use `experiments/build_phase_dataset.py` to generate a large dataset:

```bash
# Generate 150 samples per phase (600 total)
python experiments/build_phase_dataset.py --num-per-phase 150

# Custom settings
python experiments/build_phase_dataset.py \
    --num-per-phase 200 \
    --output data/my_dataset.json \
    --seed 42
```

The script randomly varies:
- System sizes: [16, 24, 32, 40]
- Circuit depths: [4, 6, 8, 10, 12]
- Bias parameters: [0.6, 0.65, 0.7, 0.75, 0.8]

### Evaluating Classifiers

Compare all three classifiers (k-NN, Centroid, MLP) using:

```bash
python experiments/evaluate_phase_classifiers.py --dataset data/phase_dataset.json
```

**Output includes:**
- Accuracy comparison table
- Mean confidence scores
- Confusion matrices for each classifier

**Expected Results:**
The MLP should achieve higher accuracy than simple k-NN or centroid methods, especially with sufficient training data.

## B. RL Lab Scientist

### What it does

The RL agent learns which experiment configurations yield high Time-Reversal Instability (TRI) scores, effectively discovering the most "interesting" quantum phases.

### Architecture

- **Environment**: `RLLabEnv`
  - **State**: Currently stateless (bandit-style), can be upgraded to include feature vectors
  - **Actions**: 36 discrete actions (4 phases × 3 biases × 3 depths)
  - **Reward**: TRI score from time-reversal experiments

- **Agent**: Simple Q-learning with ε-greedy exploration
  - **Q-table**: One value per action
  - **Learning rate (α)**: 0.1
  - **Exploration rate (ε)**: 0.2

### Running the RL Agent

```bash
# Train for 500 episodes (default)
python experiments/rl_scientist.py

# Train for more episodes
python experiments/rl_scientist.py --episodes 2000

# Use different random seed
python experiments/rl_scientist.py --episodes 1000 --seed 42
```

**Output:**
- Progress updates every 100 episodes
- Final Q-table showing expected TRI for each action
- Best action discovered (phase + bias + depth combination)

### Example Output

```
Episode 100: best_Q=2.4589
Episode 200: best_Q=3.1245
...
Episode 500: best_Q=3.8921

=== Final Q-values ===
Action  0: phase=ising_symmetry_breaking bias=0.60 depth= 4 Q=2.8945
Action  1: phase=ising_symmetry_breaking bias=0.60 depth= 8 Q=3.2156
...

Best action found:
  phase=ising_symmetry_breaking
  bias=0.7
  depth=12
  expected_TRI=3.8921
```

### Interpretation

- High Q-values indicate experiments that consistently produce large TRI scores
- The agent autonomously discovers that certain phase types and parameter combinations are more "unstable" under time-reversal
- This is essentially an AI doing science: picking experiments to maximize discovery potential

## C. Future Enhancements

### Neural Phase Classifier

1. **Add state to RL agent**:
   - Use last experiment's feature vector as state
   - Upgrade to function approximation (DQN)

2. **Different reward functions**:
   - Replay Stability Index (RSI)
   - Cluster separation
   - Entropy maximization

3. **Multi-objective RL**:
   - Optimize for both high TRI and computational efficiency
   - Balance exploration vs exploitation

### Lab LLM (Conceptual)

To build a domain-specific LLM that understands your lab:

1. **Create corpus** (`lab_corpus/`):
   - `ARCHITECTURE.md`
   - `PHASE_DETECTOR.md`
   - Discovery suite docs
   - Bit system explanations
   - Your own notes and explanations

2. **Generate instruction data**:
   - Convert docs to Q&A format (JSONL)
   - Example: `{"instruction": "Explain QPR-R", "output": "QPR-R is..."}`

3. **Fine-tune a small model**:
   - Use Unsloth/Axolotl with LoRA/PEFT
   - Models: 1.3B, 3B, or 7B parameters
   - Train on consumer GPU or cloud

4. **Export to GGUF for Ollama**:
   - Convert fine-tuned weights to GGUF format
   - Create Ollama modelfile
   - Run locally: `ollama run ben-lab-3b`

This gives you a "lab assistant" that speaks your vocabulary and can help reason about experiments.

## Integration with PhaseDetector

The MLP classifier can be used alongside the existing centroid classifier:

```python
from jarvis5090x import PhaseDetector, MLPPhaseClassifier

detector = PhaseDetector(orchestrator)

# Option 1: Use default centroid classifier (always available)
detector.train_classifier()
result = detector.classify_phase(experiment_id=exp_id)

# Option 2: Switch to MLP classifier (requires PyTorch)
if detector.mlp_classifier is not None and detector.mlp_classifier.is_available():
    detector.use_mlp_classifier()
    dataset = detector.build_dataset()
    detector.train_classifier(dataset)
    result = detector.classify_phase(experiment_id=exp_id)
else:
    print("PyTorch not available - using centroid classifier")

# Option 3: Use MLP classifier directly
mlp = MLPPhaseClassifier()
if mlp.is_available():
    dataset = detector.build_dataset()
    mlp.train(dataset)
    label, conf = mlp.predict(feature_vector)
```

## Files Created

- `jarvis5090x/phase_mlp_classifier.py` - Neural network classifier
- `experiments/build_phase_dataset.py` - Dataset generation script
- `experiments/evaluate_phase_classifiers.py` - Classifier comparison tool
- `experiments/rl_scientist.py` - RL agent for experiment optimization

## Dependencies

The core functionality works without any additional dependencies. However:

- **PyTorch** is optional for MLPPhaseClassifier neural network training
  - If PyTorch is not available, the MLP classifier will be skipped
  - The centroid and k-NN classifiers work without PyTorch
  - To enable MLP classifier: `pip install torch`

- **RL Agent** works without PyTorch (uses Q-table)

The system gracefully handles missing PyTorch and will skip MLP tests/demos automatically.

## Quick Start

```bash
# 1. Generate dataset
python experiments/build_phase_dataset.py --num-per-phase 100

# 2. Compare classifiers
python experiments/evaluate_phase_classifiers.py

# 3. Train RL agent
python experiments/rl_scientist.py --episodes 500
```

This demonstrates:
- Neural phase classification beating simple methods
- RL agent learning to choose high-reward experiments
- Your AI "student" running virtual experiments autonomously
