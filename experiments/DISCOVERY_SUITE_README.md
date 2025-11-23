# Discovery Suite: Synthetic Phase Space Exploration

Three fundamental experiments that probe the structure of your synthetic quantum phase space using the PhaseDetector and Jarvis5090X stack.

## Quick Start

```bash
python experiments/discovery_suite.py
```

## The Three Experiments

### 🔥 EXPERIMENT A: Time-Reversal Instability (TRI)

**What it measures:**
How sensitive a phase is to "arrow-of-time" perturbations in your synthetic lab.

**How it works:**
1. Run a phase experiment with bias parameter (e.g., 0.7)
2. Run a "reversed" experiment with flipped bias (1.0 - bias = 0.3)
3. Compare the feature vectors using L2 distance

**The metric: TRI (Time-Reversal Instability)**
```
TRI = ||features_forward - features_reverse||₂
```

- **Low TRI** (≈0.0001-0.001): Phase is stable under inversion - features barely change
- **High TRI** (≈0.1+): Phase is highly directional and time-fragile

**Physical intuition:**
For an Ising model, forward bias prefers +1 domains while reverse bias prefers -1 domains. If the phase is truly broken-symmetry (like Ising), you'll see large feature drift. If it's symmetric or topological, features stay similar.

**What you discover:**
- Which phases have built-in directionality vs which are time-symmetric
- Phases with high TRI are candidates for "arrow-of-time" sensitivity studies

---

### ⚡ EXPERIMENT B: Unsupervised Phase Discovery (Clustering)

**What it measures:**
The emergent structure of phase space without using ground-truth labels.

**How it works:**
1. Generate N samples per phase type (e.g., 30 × 4 phases = 120 experiments)
2. Extract feature vectors from all experiments
3. Run k-means clustering on raw features (no labels used)
4. Compare discovered clusters to known phase labels

**The metrics:**
```
Cluster purity: How many samples in a cluster share the same ground-truth label?
Cluster statistics: Label distribution within each cluster
```

**Physical intuition:**
If your feature extraction is good, phases with similar physics should cluster together naturally. You might also discover:
- Sub-phases within a single label (e.g., weakly vs strongly magnetized Ising)
- Overlaps between labels (e.g., weakly ordered phases that look similar)
- New emergent groupings your labels didn't capture

**What you discover:**
- Whether your synthetic phases form distinct families in feature space
- If your features capture meaningful physical differences
- Potential new phase boundaries your labels miss

**Example output:**
```
Cluster 0: {'ising_symmetry_breaking': 28, 'pseudorandom': 2}
Cluster 1: {'spt_cluster': 30}
Cluster 2: {'trivial_product': 30}
Cluster 3: {'pseudorandom': 28, 'ising_symmetry_breaking': 2}
```
→ Cluster 0 cleanly separates Ising phases
→ Cluster 3 finds pseudorandom phases
→ Some mixing suggests overlapping regimes

---

### 🧠 EXPERIMENT C: Replay Drift Scaling (RSI)

**What it measures:**
How fragile a phase is under parameter scaling - specifically, how features change as you increase circuit depth.

**How it works:**
1. Run base experiment with `depth = D`
2. Run same experiment with `depth = 2D, 3D, 4D, ...`
3. Measure feature drift relative to base depth

**The metric: RSI (Replay Sensitivity Index)**
```
RSI(depth) = ||features(depth) - features(base_depth)||₂
```

- **Low drift growth**: Phase is stable under scaling - "easy" phase
- **High drift growth**: Phase is chaotic under scaling - "hard" phase

**Physical intuition:**
- Stable phases (like product states) should have linear or slow drift growth
- Chaotic phases (like scrambling dynamics) might have fast drift growth
- Topological phases might show plateaus (stable features after thermalization)

**What you discover:**
- Which phases are robust to scaling vs which blow up
- Complexity hierarchy: trivial < SPT < Ising < pseudorandom (hypothesized)
- Critical depths where phases transition or thermalize

**Example output:**
```
Phase: ising_symmetry_breaking
  depth=  4  drift=0.000000
  depth=  8  drift=6.418695
  depth= 12  drift=12.728336
  depth= 16  drift=19.004699
```
→ Roughly linear growth suggests controlled scaling behavior

---

## Key Insights

### Why these experiments matter

1. **TRI**: Distinguishes directional phases from symmetric ones
   - Papers can say: "We quantified time-reversal fragility across phase families"

2. **Clustering**: Validates that your feature space has structure
   - Papers can say: "Unsupervised learning rediscovered phase boundaries without labels"

3. **RSI**: Maps complexity vs depth
   - Papers can say: "We characterized scaling behavior across the synthetic phase diagram"

### Combined discovery workflow

```
TRI → tells you WHO is time-asymmetric
Clustering → tells you HOW phases group in feature space  
RSI → tells you HOW phases behave under scaling
```

Together, they give you a **synthetic phase complexity spectrum**.

---

## Customization

### Change phases tested:
```python
tracked_phases = [
    "ising_symmetry_breaking",
    "spt_cluster", 
    "trivial_product",
    "pseudorandom",
]
```

### Adjust experiment parameters:
```python
# TRI sensitivity
run_time_reversal_test(detector, phase_type=phase, depth=16, bias=0.8)

# More samples for clustering
unsupervised_phase_discovery(detector, num_per_phase=50, k=6)

# Finer depth scaling
replay_drift_scaling(detector, depth_factors=(1, 2, 3, 4, 5, 6))
```

---

## Output Interpretation

### Strong TRI (>0.05):
Phase has built-in directionality - asymmetric under bias reversal

### Clean clustering (>90% purity):
Features strongly distinguish phases - good discriminative power

### Linear RSI growth:
Phase scales predictably - not chaotic, not trivial

### Sublinear RSI growth:
Phase saturates quickly - might be reaching thermal or trivial limit

### Superlinear RSI growth:
Phase becomes more complex with depth - scrambling or chaotic dynamics

---

## Implementation Details

- **Detector**: Uses single virtual quantum device with score=50.0
- **Feature extraction**: Automatic via PhaseDetector's built-in `extract_features`
- **Clustering**: Pure Python k-means (no sklearn dependency)
- **Reproducibility**: Uses fixed seeds for experiment A and C; random seeds for B

---

## Citation

If you use this discovery suite in your research:

```
We explored the geometry of synthetic phase space via three experiments:
(A) Time-Reversal Instability (TRI) to quantify directional sensitivity,
(B) Unsupervised clustering to discover emergent phase structure, and  
(C) Replay Drift Scaling (RSI) to characterize complexity growth with depth.
```
