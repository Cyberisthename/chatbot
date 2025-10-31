#!/usr/bin/env python3
"""
Quick Quantum Interference Experiment

Simulates a double-slit interference pattern to demonstrate wave-particle duality
and quantum interference - one of the most famous experiments in quantum mechanics.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def run_interference(n_points=2048, wavelength=1.0, slit_distance=5.0, slit_width=0.5, L=50.0):
    """
    Simulate a double-slit interference pattern.
    
    Args:
        n_points: number of detector points
        wavelength: wavelength of the simulated particle
        slit_distance: distance between slits
        slit_width: width of each slit
        L: distance from slits to detector
    
    Returns:
        dict: Results dictionary with parameters and measurements
    """
    x = np.linspace(-20, 20, n_points)
    k = 2 * np.pi / wavelength
    # amplitude from each slit
    slit1 = np.sinc((x + slit_distance / 2) / slit_width)
    slit2 = np.sinc((x - slit_distance / 2) / slit_width)
    # interference term
    intensity = (slit1 + slit2 * np.exp(1j * k * x * slit_distance / L))
    pattern = np.abs(intensity) ** 2
    pattern /= np.max(pattern)

    # save artifact
    out = {
        "experiment": "quantum_double_slit_interference",
        "version": "1.0",
        "params": {
            "n_points": n_points,
            "wavelength": wavelength,
            "slit_distance": slit_distance,
            "slit_width": slit_width,
            "L": L,
        },
        "results": {
            "mean_intensity": float(np.mean(pattern)),
            "max_intensity": float(np.max(pattern)),
            "min_intensity": float(np.min(pattern)),
            "std_intensity": float(np.std(pattern)),
        },
        "physics": {
            "phenomenon": "wave-particle duality",
            "description": "Bright and dark fringes from quantum interference",
            "wave_number": float(k),
        },
        "notes": "Classic double-slit experiment showing quantum interference patterns"
    }
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/interference_result.json", "w") as f:
        json.dump(out, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pattern, linewidth=1.5, color='blue')
    plt.title("Synthetic Quantum Interference Pattern", fontsize=14, fontweight='bold')
    plt.xlabel("Detector Position", fontsize=12)
    plt.ylabel("Normalized Intensity", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/interference_pattern.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Interference experiment complete!")
    print(f"📊 Generated artifacts/interference_pattern.png")
    print(f"📄 Generated artifacts/interference_result.json")
    
    return out


if __name__ == "__main__":
    result = run_interference()
    print("\n📈 Results:")
    print(json.dumps(result, indent=2))
