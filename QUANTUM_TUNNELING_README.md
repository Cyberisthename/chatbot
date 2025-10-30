# ⚛️ Quantum Tunneling Simulation — "Ghost Barrier Test"

A 1D quantum tunneling simulation demonstrating wave-particle behavior through potential barriers using the split-operator Fourier method.

## 🧠 Physics Summary

This experiment simulates a quantum particle approaching a potential barrier higher than its kinetic energy. Classically, the particle cannot pass through. Quantum mechanically, there's a finite probability it "tunnels" through the barrier.

### Key Physics

- **Wavefunction**: ψ(x,t) - complex probability amplitude
- **Potential**: V(x) - rectangular barrier
- **Energy**: E - particle kinetic energy
- **Transmission**: Probability of particle appearing beyond barrier
- **Reflection**: Probability of particle bouncing back

### Schrödinger Equation

```
iℏ ∂ψ/∂t = -(ℏ²/2m) ∂²ψ/∂x² + V(x)ψ
```

Solved using the split-operator method with FFT:
```
exp(-iH dt) ≈ exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2)
```

## 🚀 Quick Start

### Using the CLI

```bash
# Basic tunneling experiment (default parameters)
python3 -m quantacap.cli quantum-tunneling

# Custom parameters - high barrier (tunneling regime)
python3 -m quantacap.cli quantum-tunneling --energy 2.0 --barrier 5.0 --steps 2000

# Classical transmission (E > V)
python3 -m quantacap.cli quantum-tunneling --energy 5.0 --barrier 2.0 --steps 2000

# Without plotting
python3 -m quantacap.cli quantum-tunneling --no-plot
```

### Using Python API

```python
from quantacap.experiments.quantum_tunneling import simulate_tunneling, output_artifacts

# Run simulation
result = simulate_tunneling(
    n=1024,              # Grid points
    barrier_center=512,   # Barrier position
    barrier_width=128,    # Barrier width
    barrier_height=5.0,   # Barrier height V
    energy=2.0,          # Particle energy E
    steps=2000,          # Time steps
    dt=0.002,            # Time step size
)

print(f"Transmission: {result.final_transmission:.6f}")
print(f"Reflection:   {result.final_reflection:.6f}")

# Save artifacts
artifacts = output_artifacts(
    result,
    out="artifacts/tunneling_result.json",
    plot="artifacts/tunneling_evolution.png",
)
```

### Quick Demo

```bash
python3 test_quantum_tunneling.py
```

## 📊 Output Artifacts

### JSON Results

Location: `artifacts/tunneling_result.json`

```json
{
  "experiment": "quantum_tunneling",
  "params": {
    "energy": 2.0,
    "barrier_height": 5.0,
    "barrier_width": 128.0,
    "steps": 2000.0
  },
  "results": {
    "final_transmission": 0.000407,
    "final_reflection": 0.954486,
    "steps": 2000,
    "transmission_history": [...],
    "reflection_history": [...],
    "total_probability_history": [...]
  }
}
```

### Visualization

Location: `artifacts/tunneling_evolution.png`

The plot shows:
- **Top panel**: Time evolution of transmission (blue), reflection (red), and total probability (black dashed)
- **Bottom panel**: Transmission ratio over time with final value indicated

## 🧪 CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--energy` | 2.0 | Particle kinetic energy |
| `--barrier` | 5.0 | Potential barrier height |
| `--steps` | 2000 | Number of time evolution steps |
| `--n` | 1024 | Number of spatial grid points |
| `--barrier-width` | 128 | Barrier width in grid points |
| `--dt` | 0.002 | Time step size |
| `--seed` | 424242 | Random seed |
| `--out` | `artifacts/tunneling_result.json` | Output JSON path |
| `--plot-prefix` | `artifacts/tunneling_evolution` | Plot file prefix |
| `--plot` / `--no-plot` | `--plot` | Generate visualization |

## 🔬 Physical Interpretation

### Tunneling Regime (E < V)

When the particle energy is **less than** the barrier height:
```bash
python3 -m quantacap.cli quantum-tunneling --energy 2.0 --barrier 5.0
```
- **Result**: ~0.04% transmission
- **Interpretation**: Quantum tunneling effect - particle has small probability to appear beyond the barrier despite being classically forbidden

### Classical Regime (E > V)

When the particle energy is **greater than** the barrier height:
```bash
python3 -m quantacap.cli quantum-tunneling --energy 5.0 --barrier 2.0
```
- **Result**: ~53% transmission
- **Interpretation**: Classical transmission - particle has enough energy to pass over the barrier

### Conservation Laws

The simulation conserves probability:
```
|ψ|² integrated over all space ≈ 1.0
Transmission + Reflection + (barrier region) ≈ 1.0
```

## 🧪 Running Tests

```bash
cd quantacap
python3 -m pytest tests/test_quantum_tunneling.py -v
```

Tests verify:
- Probability conservation (total ≈ 1.0)
- Low transmission for high barriers (E < V)
- High transmission for low barriers (E > V)
- Parameter validation
- Result data structure

## 🎓 Educational Value

This experiment demonstrates:

1. **Wave-particle duality**: Particle described by wavefunction
2. **Quantum tunneling**: Penetration through classically forbidden regions
3. **Probability interpretation**: |ψ|² gives position probability
4. **Numerical methods**: Split-operator FFT method for PDE solving
5. **Conservation laws**: Total probability conservation

## 📚 Further Reading

- Quantum Mechanics: Griffiths, "Introduction to Quantum Mechanics"
- Numerical Methods: Press et al., "Numerical Recipes"
- Split-Operator Method: Feit & Fleck (1982)

## 🔧 Implementation Details

- **Method**: Split-operator Fourier method (spectral accuracy)
- **Units**: Natural units (ℏ=1, m=1/2)
- **Normalization**: Wave function renormalized each step
- **FFT**: NumPy's FFT for efficient momentum-space evolution
- **Barrier**: Rectangular potential (can be extended)

## 🎯 Example Results

| Scenario | Energy (E) | Barrier (V) | Transmission | Physical Regime |
|----------|-----------|-------------|--------------|-----------------|
| Tunneling | 2.0 | 5.0 | 0.04% | Quantum (E<V) |
| Mixed | 4.0 | 4.5 | ~5% | Transition |
| Classical | 5.0 | 2.0 | 53% | Classical (E>V) |

The dramatic difference in transmission probability demonstrates the quantum nature of matter!
