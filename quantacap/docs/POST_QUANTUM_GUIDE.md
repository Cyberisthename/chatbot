# Post-Quantum Upgrade Pack Guide

The Post-Quantum (PQ) Upgrade Pack is a collection of six exploratory computational toy models inspired by exotic physics concepts. These are **not** claims of surpassing quantum mechanics, but rather experimental frameworks for exploring alternative computational paradigms.

## Overview

| Track | Concept | Key Metric | Physics Inspiration |
|-------|---------|------------|-------------------|
| **fields** | Sub-quantum field interference | Visibility, Mutual Information | Complex Ginzburg-Landau, Wave Optics |
| **topo** | Topological braid logic | Fidelity, Stability | Anyon braiding, Topological QC |
| **relativity** | Spacetime computing | Speedup (proper time vs Newtonian) | Special/General Relativity |
| **holo** | Holographic entropy bounds | Area-law scaling (H ∝ A) | Holographic principle, Black hole thermodynamics |
| **biotoy** | Adaptive neural matter | PSNR, Memory halftime | Reaction-diffusion, Hebbian plasticity |
| **hyperdim** | Tensor network compression | Overlap, Memory scaling | Matrix Product States, Quantum tensor networks |

---

## 1. Fields — Sub-Quantum Field Computing

### Physical Intuition

Computing with continuous complex fields φ(x,y,t) on a 2D grid. Sources inject phase patterns, and logic emerges from **interference** at detector locations. Inspired by analog wave computing and continuous-field models.

### What's Toy vs Real

- **Toy**: Simplified linear wave update with damping, no true QFT.
- **Real mapping**: Relates to optical computing, phase-locked loops, and interference-based logic gates.

### Key Metrics

- **Visibility** = (I_max - I_min) / (I_max + I_min) at detectors, where I = |φ|².  
  High visibility → strong interference patterns.
- **Mutual Information**: Entropy-based measure of how input phase patterns correlate with output detector readings.
- **Energy**: Total field intensity ∫|φ|²dx dy.

### Parameters

- `--N`: Grid size (larger = finer spatial resolution, slower).
- `--T`: Time evolution steps (longer = more stable patterns).
- `--src`: Number of phase sources injecting signals.
- `--seed`: Deterministic random seed.
- `--gif`: Generate animated GIF of field evolution.

### Example

```bash
python -m quantacap.cli pq-fields --N 256 --T 400 --src 2 --gif
```

**Artifacts**:
- `fields_summary.json`: visibility, mutual info, energy
- `fields_last.npy`: final complex field snapshot
- `fields_interf.png`: spatial interference pattern with marked sources/detectors
- `fields_evolution.gif` (if --gif): animated evolution

### Interpretation

If visibility > 0.7 and mutual info > 1.0 bit, the field demonstrates robust interference-based information encoding.

---

## 2. Topo — Topological Braid Logic

### Physical Intuition

Anyonic worldlines in 2+1D spacetime form braids. Braid operations map to unitary gates on a small qubit register. **Topological protection**: local noise (path jitter) shouldn't change the braid class, so computation is robust.

### What's Toy vs Real

- **Toy**: Discrete braid word parsed into 2×2/4×4 unitary generators; noise is added then re-projected to unitary form.
- **Real mapping**: Topological quantum computing with Majorana fermions or fractional quantum Hall anyons.

### Key Metrics

- **Fidelity**: overlap |⟨ψ_clean | ψ_noisy⟩|² after noise perturbation.
- **Topo Stability**: 1 - mean(||U_noisy - U_clean|| / sqrt(size)). Closer to 1 → robust to noise.
- **Braid Length**: number of generators in the word.

### Parameters

- `--braid`: Braid word string (e.g., `"s1 s2^-1 s1"`). `s1`, `s2` are generators; `^-1` denotes inverse.
- `--shots`: Number of noisy trials.
- `--noise`: Amplitude of Gaussian noise added to unitary matrix.
- `--seed`: Random seed.

### Example

```bash
python -m quantacap.cli pq-topo --braid "s1 s2^-1 s1" --shots 8192 --noise 0.03
```

**Artifacts**:
- `topo_summary.json`: fidelity, topo_stability, braid_length
- `unitary.npy`: clean braid unitary
- `braid_plot.png`: anyon worldline trajectories
- `histogram.png`: measurement outcome distribution

### Interpretation

High fidelity (> 0.95) and topo_stability (> 0.98) with moderate noise indicate robust "knot logic."

---

## 3. Relativity — Spacetime / Relativistic Computing

### Physical Intuition

A directed task graph where each node has a local clock running at γ = 1/√(1 - v²/c²). Tasks on "faster-moving" nodes experience time dilation, finishing sooner in coordinate time. We compute the **proper-time** path length and compare to Newtonian time.

### What's Toy vs Real

- **Toy**: DAG with random velocities; simplified Lorentz factor applied to edge weights.
- **Real mapping**: Relativistic distributed computing, gravitational time dilation near massive objects (GPS corrections).

### Key Metrics

- **Speedup**: ratio of Newtonian completion time to relativistic proper-time completion.
- **γ-statistics**: mean, max, min of Lorentz factors across nodes.
- **Causality Violations**: must be zero (DAG ensures this).

### Parameters

- `--nodes`: Number of computational tasks.
- `--edges`: Number of dependencies.
- `--beta`: Maximum velocity as fraction of c (0 < β < 1).
- `--seed`: Random seed.

### Example

```bash
python -m quantacap.cli pq-relativity --nodes 64 --edges 256 --beta 0.6
```

**Artifacts**:
- `relativity_summary.json`: speedup, γ-stats, critical path
- `timing_hist.png`: Newtonian vs relativistic time profiles
- `graph.png`: task graph with critical path highlighted

### Interpretation

Speedup > 1 means relativistic effects provide computational advantage. Typical values: 1.1–1.5× for β ~ 0.6.

---

## 4. Holo — Holographic / Entropy Computing

### Physical Intuition

The **holographic principle** suggests information in a volume is bounded by its surface area: I_max ∝ A. We measure Shannon entropy H(r) inside random regions and verify H ≈ k·A(r) with linear fit.

### What's Toy vs Real

- **Toy**: Binary 3D voxel grid; Shannon entropy of random subregions.
- **Real mapping**: Bekenstein bound, AdS/CFT correspondence, black hole entropy S = A/(4G).

### Key Metrics

- **k_fit**: slope of H(r) vs A(r) linear fit.
- **R²**: goodness of fit (1.0 = perfect).
- **Holo Ratio**: H(r) / (k·A(r)), should be ~ 1 if holographic scaling holds.
- **Residual Std**: scatter around the fit line.

### Parameters

- `--N`: Voxel grid size (NxNxN).
- `--samples`: Number of random regions to sample.
- `--seed`: Random seed.

### Example

```bash
python -m quantacap.cli pq-holo --N 64 --samples 50
```

**Artifacts**:
- `holo_summary.json`: k_fit, R², residuals
- `H_vs_area.png`: scatter plot with linear fit
- `density_slice.png`: mid-plane voxel density

### Interpretation

R² > 0.9 and small residual_std confirm approximate holographic behavior. Deviations suggest "bulk" effects or insufficient sampling.

---

## 5. BioToy — Bio / Consciousness Computing

### Physical Intuition

A 2D neural field φ(x,y,t) coupled to a plastic connectivity matrix W(t). The system learns to reproduce a target spatiotemporal pattern P(x,y,t) via Hebbian-like updates, subject to an energy budget E = ∑φ² + λ‖W‖².

After training, we test "dream replay": turn off input and observe pattern persistence.

### What's Toy vs Real

- **Toy**: Reaction-diffusion + Hebbian plasticity; simplified neural dynamics.
- **Real mapping**: Neural oscillator networks, cortical columns, wetware memory consolidation.

### Key Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): reconstruction quality of φ vs target P.  
  PSNR = 20·log₁₀(max|P|) - 10·log₁₀(MSE).  
  Higher = better match.
- **Energy**: mean(E) over final window, measures metabolic cost.
- **Memory Halftime**: time in dream replay until correlation with target drops to 50%.

### Parameters

- `--N`: Grid size (NxN).
- `--T`: Training steps.
- `--lambda`: Regularization strength (higher = sparser weights).
- `--seed`: Random seed.
- `--gif`: Generate dream replay GIF.

### Example

```bash
python -m quantacap.cli pq-biotoy --N 128 --T 500 --lambda 0.01 --gif
```

**Artifacts**:
- `biotoy_summary.json`: PSNR, energy, halftime
- `energy_curve.png`: energy trajectory over training
- `biotoy_replay.gif` (if --gif): dream replay dynamics

### Interpretation

PSNR > 30 dB and low energy indicate efficient learning. Long halftime (> 5.0) suggests stable memory encoding.

---

## 6. Hyperdim — Hyperdimensional / Tensor-Network Computing

### Physical Intuition

Represent N-qubit states as **Matrix Product States (MPS)** with bond dimension χ. Apply random quantum circuits and measure overlap with exact dense simulation (for small N). Trade off accuracy vs memory: larger χ = higher accuracy but more memory.

### What's Toy vs Real

- **Toy**: Random single-qubit rotations + two-qubit entanglers; SVD truncation to bond χ.
- **Real mapping**: Quantum tensor networks (DMRG, TEBD), quantum simulation on classical hardware.

### Key Metrics

- **Overlap**: |⟨ψ_dense | ψ_MPS⟩|². Closer to 1 = MPS accurately represents state.
- **Memory (bytes)**: total memory for MPS tensors.
- **Runtime (seconds)**: MPS simulation time.
- **Bond Dimensions**: list of bond dimensions after each two-site gate.

### Parameters

- `--N`: Number of qubits/sites.
- `--chi`: Maximum bond dimension (truncation threshold).
- `--depth`: Circuit depth (layers of gates).
- `--seed`: Random seed.

### Example

```bash
python -m quantacap.cli pq-hyperdim --N 48 --chi 32 --depth 40
```

**Artifacts**:
- `hyperdim_summary.json`: overlap, memory, runtime, bond_dims
- `accuracy_vs_chi.png`: plot of overlap vs χ for reference system

### Interpretation

Overlap > 0.99 with χ = 32 shows efficient compression. Memory scales ~ O(N·χ²·d) vs dense O(2^N·d).

---

## Reproducing Figures

Each experiment generates summary JSON and at least one PNG. For GIFs, add `--gif` flag to `pq-fields` and `pq-biotoy`.

**Batch run all experiments**:
```bash
for cmd in pq-fields pq-topo pq-relativity pq-holo pq-biotoy pq-hyperdim; do
  python -m quantacap.cli $cmd
done
```

All artifacts save to `artifacts/pq/<track>/`.

---

## Ethics & Scope Note

These experiments are **exploration tools**, not claims of new physics or quantum advantage. They:

- Illustrate alternative computational paradigms inspired by theoretical physics.
- Serve as testbeds for algorithmic ideas (interference logic, topological robustness, etc.).
- **Do NOT** replace or challenge established quantum mechanics.

Use responsibly for education, research prototyping, and conceptual exploration.

---

## Troubleshooting

### ImportError: No module named 'matplotlib'
- Install with: `pip install matplotlib`
- Experiments will skip plot generation gracefully if matplotlib unavailable.

### Slow execution with large N or T
- Reduce grid size (`--N`) or time steps (`--T`).
- For `pq-hyperdim`, reduce `--depth` or `--chi`.

### NaN or inf in metrics
- Check seed; some random configurations may produce degenerate cases.
- Increase regularization (`--lambda` in biotoy) or decrease noise (`--noise` in topo).

---

## References

- **Fields**: Wave-based computing, optical interference gates
- **Topo**: Kitaev anyons, topological quantum computation (Freedman et al.)
- **Relativity**: Relativistic task scheduling, GPS time dilation
- **Holo**: Bekenstein bound, 't Hooft holographic principle, AdS/CFT
- **BioToy**: Hebbian learning, cortical neural fields, Integrated Information Theory
- **Hyperdim**: Matrix Product States (Schollwöck), TEBD, quantum tensor networks

---

**Version**: 1.0  
**Maintainer**: Quantacap Team  
**License**: See repository LICENSE
