# Ben Lab Bit System - Synthetic Quantum Primitives

## Overview

Ben Lab experiments rely on a custom vocabulary of synthetic "bits" that extend classical qubits with replay, biasing, and memory primitives. These bits live inside adapters, are fully deterministic, and can be replayed exactly—perfect for the Jarvis-5090X phase lab.

---

## X-bit (Baseline Adapter Qubit)
The foundational amplitude pair used across the lab.

**Purpose:**
- Represents the canonical |0⟩/|1⟩ amplitude state stored in adapters
- Provides normalized α/β amplitudes before any biasing
- Acts as the starting point for constructing more exotic bits

**Properties:**
- Deterministic superposition: `(α, β)` with |α|² + |β|² = 1
- Supports replay via adapter snapshots
- Used by PhaseDetector generators when building initial branch states

---

## Y-bit (Hybrid Qubit ⊗ Z-bias)
A hybrid qubit with phase nudge combining X-bit amplitudes with Z-bit bias.

**Properties:**
- Combines qubit superposition with Z-bit continuum bias
- Phase adjustments via ε-phase tied to the Z-bit
- Used in CHSH-Y experiments to tilt measurement frames
- Deterministic and replayable

**Usage:**
```python
# Y-bit enables measurement frame adjustments in CHSH experiments
# Preserves quantum structure while exposing new patterns
```

---

## Z-bit (Continuum Bias Scalar)
A scalar defined over the continuum excluding [1,2].

**Properties:**
- Continuous domain: ℝ \ [1,2]
- Supplies bias values for phase experiments and replay tests
- Allows fine-grained control of directionality (e.g., bias=0.7)

**Usage in Phase Experiments:**
```python
# Forward experiment: bias = 0.7 (Z-bit domain)
# Reverse experiment: bias = 1.0 - 0.7 = 0.3
# Used in TRI (Time-Reversal Instability) measurements
```

---

## A-bit (Amplitude Memory Cell)
Captures amplitude history during an experiment run.

**Purpose:**
- Caches amplitude evolutions per layer for replay
- Provides hooks for "never recompute" inference in Jarvis-5090X
- Used when exporting feature vectors or training classifiers

**Properties:**
- Stores sequences of amplitude snapshots
- Deterministic serialization inside adapter logs
- Allows the RL scientist to revisit amplitude branches mid-training

---

## S-bit (Scrambling Indicator)
Tracks how uniformly a branch diffuses probability mass.

**Purpose:**
- Mirrors the `scrambling_score` metric in feature vectors
- Flags when a phase approaches pseudorandom behavior
- Drives decisions inside Discovery Suite clustering experiments

**Properties:**
- Derived from branch entropy statistics
- 0 ≈ highly ordered, 1 ≈ fully scrambled
- Stored side-by-side with A-bit amplitude logs

---

## T-bit (Time-Phase Pointer)
Encodes temporal offsets and time-reversal markers.

**Purpose:**
- Marks forward vs reverse passes in TRI experiments
- Keeps track of layer depth when computing RSI curves
- Allows deterministic reversal (seeded timing) during replay

**Properties:**
- Contains discrete timestep index + bias direction flag
- Enables `PhaseDetector.replay_experiment` to align logs
- Used by the Quantum Approximation Layer when collapsing branches

---

## C-bit (Correlation Carrier)
Tracks correlation structure across branches and subsystems.

**Purpose:**
- Backs the `correlation_*` entries of the feature vector
- Records pairwise mutual information proxies during simulation
- Supports clustering by exposing phase-specific correlation fingerprints

**Properties:**
- Stores rolling averages and extrema of correlations
- Normalized to [0,1] for comparability across depths
- Interfaces with the Phase MLP inputs when training classifiers

---

## P-bit (Path Memory)
Keeps deterministic paths through adapter graphs.

**Purpose:**
- Records the exact branch path chosen during collapse
- Enables 1:1 replay of stochastic-looking processes (actually deterministic)
- Forms the basis for RL scientist trajectory storage

**Properties:**
- Encodes branch indices + random seeds
- Allows "branch stitching" when rehydrating experiments
- Critical for RL exploration vs exploitation analysis

---

## R-bit (Replay Anchor)
Guarantees identical re-execution of synthetic experiments.

**Purpose:**
- Serves as the pointer into adapter storage for replays
- Ties together X/Y/Z/A/S/T/C/P data for deterministic retrieval
- Fundamental to QPR-R (Quantum Phase Recognition with Replay)

**Properties:**
- Hash-based pointer referencing canonical adapter snapshots
- Used by `PhaseDetector.replay_experiment`
- Ensures that feature extraction, classifier training, and evaluation all align

---

## G-graph (Convergent Adapter Weave)
A convergent, decaying weave over thousands of adapters—"fall of infinity."

**Purpose:**
- Provides large-scale structure for P-bit path routing
- Supports networked experiments (e.g., clustering, RL exploration)
- Supplies amplitude damping factors for branch influence summaries

**Properties:**
- Deterministic graph with tunable branching factor and decay γ
- `influence(seed)` returns two probabilities used to seed Y-bit adjustments
- Allows synthetic networks to scale while remaining replayable

---

## Adapter Architecture
Adapters are the fundamental storage/replay units in the Ben Lab.

**Key Features:**
- Deterministic storage of quantum-like states
- Instant replay capability (driven by R-bit anchors)
- Used across CHSH experiments, atom simulations, and phase logging
- Offline, synthetic, replayable—no quantum randomness required

---

## Bit System in Practice

### In Phase Detection
- **Z-bit** controls bias in Ising symmetry breaking (magnetization direction)
- **T-bit** marks forward vs reverse passes for TRI
- **C-bit/S-bit** feed feature vector correlation and scrambling metrics
- **R-bit** guarantees replay-aligned training data

### In Discovery Suite
- **Z-bit** drives TRI reversal (bias → 1 - bias)
- **S-bit/C-bit** enable unsupervised clustering to spot emergent structure
- **T-bit** and **A-bit** underpin RSI depth sweeps
- **G-graph** orchestrates large sampling runs without losing determinism

### In Quantum Approximation Layer
- **X-bit** seeds branch amplitudes
- **Y-bit** applies bias-induced phase nudges
- **P-bit** records branch selection order
- **R-bit** links collapse outputs back to adapter storage

---

## Physical Interpretation

**Synthetic, Not Physical:**
All bits are simulated—classical compute producing quantum-like patterns.

**Deterministic:**
Same input → same adapter → same replay. No hidden randomness.

**Research Tool:**
Enables hypothesis generation, automated experiment design, and pattern exploration before committing to expensive physical hardware.
