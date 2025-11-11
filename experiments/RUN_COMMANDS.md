# Quick Run Commands for All Experiments

## 3D Atom Solver - Discovery Mode (NEW!)

Run the progressive 3D Schrödinger solver:

```bash
# Method 1: Direct execution
python -m experiments.solve_atom_3d_discovery

# Method 2: Via CLI
python cli.py atom-3d-discovery

# Quick test (32³, 50 steps)
python test_atom_3d_discovery.py
```

**Output:** `artifacts/real_atom_3d/`
- 3D volumes at 64³, 128³, 256³
- Max-intensity projections (XY, XZ, YZ views)
- Energy convergence data
- Complete metadata JSON

**Time:** ~20 minutes for all stages

---

## Adapter Double-Slit Experiment

Run the digital double-slit interference experiment:

```bash
# Method 1: Direct execution
python -m experiments.adapter_double_slit

# Method 2: Via CLI
python cli.py adapter-double-slit
```

**Output:** `artifacts/adapter_double_slit/`
- Interference and control patterns
- Visibility analysis

**Time:** < 1 minute

---

## Original Atom Solver

Run the single-resolution atom solver:

```bash
# Via CLI
python cli.py atom-from-constants

# Or
python cli.py solve-atom
```

**Output:** `artifacts/real_atom/`
- 64³ resolution
- 2D slices and MIP
- Convergence plot

**Time:** ~1 minute

---

## View Results

```bash
# List all artifacts
ls -R artifacts/

# View a specific experiment
ls -lh artifacts/real_atom_3d/

# Check JSON metadata
cat artifacts/real_atom_3d/atom3d_descriptor.json | head -30

# View images (if on a system with image viewer)
# artifacts/real_atom_3d/atom_mip_xy.png
# artifacts/real_atom_3d/atom_mip_xz.png
# artifacts/real_atom_3d/atom_mip_yz.png
```

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key packages:
- `numpy` - Required for all experiments
- `matplotlib` - Required for visualization
- `scipy` - Optional, for better upsampling
- `imageio` - Optional, for spin GIF export

---

## Experiment Comparison

| Feature | adapter_double_slit | solve_atom_from_constants | solve_atom_3d_discovery |
|---------|-------------------|-------------------------|------------------------|
| Purpose | Quantum interference | 3D atom (single res) | 3D atom (progressive) |
| Grid | 1D (512 points) | 3D (64³) | 3D (64³→128³→256³) |
| Time | < 1 min | ~1 min | ~20 min |
| Output size | ~1 MB | ~10 MB | ~440 MB |
| Resolution | N/A | Fixed | Progressive |
| Views | 2 (interference + control) | 5 slices + 1 MIP | 3 orthogonal MIPs |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

```bash
pip install numpy matplotlib
```

### "MemoryError" at N=256³

The solver will save N=128³ results and stop gracefully.
To prevent this, modify the config:

```python
# In experiments/solve_atom_3d_discovery.py
CONFIG["N_stages"] = [64, 128]  # Skip 256³
```

### Energy diverging

Reduce the time step:

```python
CONFIG["dt"] = 0.001  # Smaller = more stable
```

---

## For Developers

### Run tests

```bash
python test_atom_3d_discovery.py
```

### Check what's installed

```bash
python -c "import numpy; import matplotlib; import scipy; import imageio; print('All packages available')"
```

### Clean artifacts

```bash
rm -rf artifacts/real_atom_3d/
```

---

## Documentation

- `experiments/ATOM_3D_DISCOVERY_README.md` - Detailed guide
- `ATOM_3D_DISCOVERY_SUMMARY.md` - Implementation summary
- `experiments/README.md` - General experiments guide
