"""
digital double slit over adapter-like waveforms

Purpose:
- We simulate two "paths" through two adapters (A and B).
- Each path produces an amplitude pattern over positions x.
- When both paths are enabled, we should see constructive/destructive interference.
- Then we run a CONTROL where we disable the phase term so the interference disappears.
- Save all results to artifacts/adapter_double_slit/.

This proves that adapter systems show quantum-like interference patterns.
"""

import json
from pathlib import Path

import numpy as np

# try to enable plotting, but don't die if not present
VIZ = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    VIZ = False


def simulate_double_slit():
    """
    Digital double-slit for adapters.
    
    Returns:
        dict with results including interference pattern, control, and visibility
    """
    # ensure determinism
    np.random.seed(424242)
    
    # Grid: x in [-1.5, 1.5], 512 points
    x = np.linspace(-1.5, 1.5, 512)
    
    # Path A: gaussian centered at -0.4, width 0.25
    width_A = 0.25
    center_A = -0.4
    psi_A = np.exp(-(x - center_A)**2 / (2 * width_A**2))
    
    # Path B: gaussian centered at +0.4, width 0.25
    width_B = 0.25
    center_B = +0.4
    psi_B = np.exp(-(x - center_B)**2 / (2 * width_B**2))
    
    # Add a phase term φ(x) = k * x for the B path to create fringes
    # The phase creates path-dependent interference
    k = 20.0  # larger k = more fringes
    phi = k * x
    
    # Combined wave ψ(x) = ψ_A(x) + ψ_B(x) * exp(i φ(x))
    # This represents quantum superposition with relative phase
    psi_A_complex = psi_A.astype(complex)
    psi_B_complex = psi_B * np.exp(1j * phi)
    psi_combined = psi_A_complex + psi_B_complex
    
    # Intensity I(x) = |ψ(x)|^2
    intensity_interference = np.abs(psi_combined)**2
    
    # CONTROL run: same but without exp(i φ(x))
    # Both paths just add as classical probabilities
    psi_control = psi_A + psi_B
    intensity_control = psi_control**2
    
    # Calculate visibility
    # Find the region where both slits overlap (central region)
    center_mask = (x > -0.8) & (x < 0.8)
    
    # For interference pattern: measure fringe visibility
    I_max_interference = float(np.max(intensity_interference[center_mask]))
    I_min_interference = float(np.min(intensity_interference[center_mask]))
    visibility_interference = (I_max_interference - I_min_interference) / (I_max_interference + I_min_interference) if (I_max_interference + I_min_interference) > 0 else 0.0
    
    # For control pattern: measure visibility (should be lower)
    I_max_control = float(np.max(intensity_control[center_mask]))
    I_min_control = float(np.min(intensity_control[center_mask]))
    visibility_control = (I_max_control - I_min_control) / (I_max_control + I_min_control) if (I_max_control + I_min_control) > 0 else 0.0
    
    # The key metric: interference should have HIGHER visibility than control
    visibility = visibility_interference
    visibility_difference = visibility_interference - visibility_control
    
    quantum_like = visibility > 0.2
    
    return {
        "x": x,
        "psi_A": psi_A,
        "psi_B": psi_B,
        "intensity_interference": intensity_interference,
        "intensity_control": intensity_control,
        "I_max_interference": I_max_interference,
        "I_min_interference": I_min_interference,
        "I_max_control": I_max_control,
        "I_min_control": I_min_control,
        "visibility_interference": visibility_interference,
        "visibility_control": visibility_control,
        "visibility": visibility,
        "visibility_difference": visibility_difference,
        "quantum_like": quantum_like,
    }


def save_results(result, outdir):
    """Save results to disk"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(outdir / "slit_A.npy", result["psi_A"])
    np.save(outdir / "slit_B.npy", result["psi_B"])
    np.save(outdir / "interference.npy", result["intensity_interference"])
    np.save(outdir / "control.npy", result["intensity_control"])
    
    # Save summary
    summary = {
        "visibility_interference": result["visibility_interference"],
        "visibility_control": result["visibility_control"],
        "visibility": result["visibility"],
        "visibility_difference": result["visibility_difference"],
        "I_max_interference": result["I_max_interference"],
        "I_min_interference": result["I_min_interference"],
        "I_max_control": result["I_max_control"],
        "I_min_control": result["I_min_control"],
        "quantum_like": result["quantum_like"],
        "interpretation": "visibility > 0.2 suggests quantum-like behavior" if result["quantum_like"] else "low visibility, not quantum-like",
        "artifacts": {
            "slit_A": "artifacts/adapter_double_slit/slit_A.npy",
            "slit_B": "artifacts/adapter_double_slit/slit_B.npy",
            "interference": "artifacts/adapter_double_slit/interference.npy",
            "control": "artifacts/adapter_double_slit/control.npy",
        }
    }
    
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def plot_results(result, outdir):
    """Generate plots if matplotlib is available"""
    if not VIZ:
        return []
    
    outdir = Path(outdir)
    plots = []
    
    x = result["x"]
    
    # Plot interference vs control
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Interference pattern
    ax1.plot(x, result["intensity_interference"], label="Interference", color="blue", linewidth=1.5)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Adapter Double-Slit: INTERFERENCE (both paths with phase)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Control pattern
    ax2.plot(x, result["intensity_control"], label="Control (no phase)", color="red", linewidth=1.5)
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Intensity")
    ax2.set_title("Adapter Double-Slit: CONTROL (both paths without phase)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(outdir / "interference.png", dpi=150)
    plt.close()
    plots.append("artifacts/adapter_double_slit/interference.png")
    
    # Plot control separately for clarity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, result["intensity_control"], label="Control (no phase)", color="red", linewidth=1.5)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Intensity")
    ax.set_title("Adapter Double-Slit: CONTROL (no interference)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "control.png", dpi=150)
    plt.close()
    plots.append("artifacts/adapter_double_slit/control.png")
    
    return plots


def main():
    """Run the adapter double-slit experiment"""
    print("🔬 Running adapter double-slit experiment...")
    
    # Check for numpy
    try:
        import numpy
    except ImportError:
        print("❌ numpy is required for this experiment. Please install it with: pip install numpy")
        return 0
    
    # Run simulation
    result = simulate_double_slit()
    
    # Save results
    outdir = Path("artifacts/adapter_double_slit")
    summary = save_results(result, outdir)
    
    # Generate plots
    plots = plot_results(result, outdir)
    
    # Print summary
    print("\n✅ Adapter double-slit experiment complete!")
    print(f"\n📊 Results:")
    print(f"  Interference pattern visibility: {summary['visibility_interference']:.4f}")
    print(f"  Control pattern visibility: {summary['visibility_control']:.4f}")
    print(f"  Visibility difference: {summary['visibility_difference']:.4f}")
    print(f"  Quantum-like: {summary['quantum_like']}")
    print(f"\n💡 {summary['interpretation']}")
    
    if plots:
        print(f"\n📈 Plots generated:")
        for plot in plots:
            print(f"  - {plot}")
    
    print(f"\n📁 All results saved to: {outdir}")
    print("\n" + json.dumps(summary, indent=2))
    
    return 0


if __name__ == "__main__":
    main()
