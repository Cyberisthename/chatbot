# Infinite Compute Lab – Unified Streamlit App

A unified physics simulation interface combining multiple experiments in one clean Streamlit application.

## Features

### 1. **Atom 3D** - 3D Schrödinger Solver
- True imaginary-time evolution solver (no precomputed orbitals)
- Solves `-½∇²ψ + V(r)ψ = Eψ` on 3D grid
- Shows 3 MIP (max-intensity projections) views
- Radial density comparison with analytic 1s orbital
- Download options: `density.npy`, `psi.npy`, `energy.json`

### 2. **Double-Slit** - Adapter Interference
- Quantum-like interference patterns
- Visibility readout
- Control vs interference comparison
- Adjustable fringe parameter

### 3. **Field Interference** - 2D Wave Dynamics
- Complex field evolution with multiple sources
- Detector intensity tracking
- Visibility measurements
- Real-time simulation

### 4. **CHSH Bell Inequality**
- Quantum entanglement test
- S-parameter calculation (S > 2 = quantum violation)
- Configurable noise and shots
- Bell state simulation

### 5. **Relativistic Task Graph**
- Time dilation in parallel computation
- Newtonian vs relativistic proper time comparison
- DAG-based task scheduling
- Adjustable velocity β

### 6. **Holographic Entropy**
- Toy model of area-law entropy scaling
- 3D voxel field generation
- Entropy ~ Area relationship
- Linear fit with visualization

## Installation

```bash
pip install streamlit numpy matplotlib scipy
```

## Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

## Features

- **Deterministic**: All simulations use fixed seed 424242
- **Local execution**: Runs entirely on CPU, no external dependencies
- **Interactive**: Real-time parameter adjustment with sliders
- **Visual**: Matplotlib plots embedded in each tab
- **Download**: Export results for further analysis

## Architecture

The app is structured as a thin wrapper around core simulation functions:
- Each tab contains one experiment
- Sliders control key parameters
- Run buttons trigger computations
- Results display as plots and metrics
- Download buttons for data export

## Notes

- All math runs locally on CPU
- Deterministic seed ensures reproducible results
- No GPU required (though some experiments benefit from it)
- Suitable for educational and research purposes

© 2025 Infinite Compute Lab
