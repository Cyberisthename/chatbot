#!/usr/bin/env python3
"""
Quantum-Style Uncertainty Collapse Experiment

Shows how random noise (chaos) gradually collapses into a stable quantum-like
amplitude distribution — an easy visual "quantum stabilization" demo.

This is a digital analog of decoherence — how chaos (quantum randomness)
stabilizes into classical order. The animation shows how amplitude² converges
to a normal distribution while losing phase variation.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Experiment parameters
n_points = 256
frames = 200
decay = 0.98   # how fast randomness stabilizes
amplitude = 1.0

# Setup
x = np.linspace(-4, 4, n_points)
psi = np.exp(-x**2)
noise = np.random.randn(frames, n_points)

fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2, color='cyan')
ax.set_ylim(0, 1)
ax.set_title("Synthetic Quantum Stability Experiment")
ax.set_xlabel("Position")
ax.set_ylabel("Amplitude²")

# Update function
def update(frame):
    global psi, amplitude
    phase = np.exp(1j * (frame / 10))
    psi = np.exp(-x**2) * phase + amplitude * noise[frame] * 0.1
    amplitude *= decay
    prob = np.abs(psi)**2
    line.set_ydata(prob / np.max(prob))
    ax.set_title(f"Frame {frame} | randomness={amplitude:.3f}")
    return line,

# Log final results as artifacts
def save_artifacts():
    prob_final = np.abs(psi)**2
    
    results = {
        "experiment": "quantum_uncertainty_collapse",
        "n_points": n_points,
        "frames": frames,
        "decay_rate": decay,
        "initial_amplitude": 1.0,
        "final_amplitude": float(amplitude),
        "final_entropy": float(np.var(prob_final)),
        "final_mean": float(np.mean(prob_final)),
        "final_std": float(np.std(prob_final)),
        "position_mean": float(np.sum(x * prob_final) / np.sum(prob_final)),
        "position_std": float(np.sqrt(np.sum(x**2 * prob_final) / np.sum(prob_final))),
        "notes": "Simulates decoherence: chaos (quantum randomness) stabilizing into classical order"
    }
    
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    output_path = artifacts_dir / "uncertainty_experiment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Artifacts saved to {output_path}")
    print(f"Final entropy (variance): {results['final_entropy']:.6f}")
    print(f"Final amplitude: {results['final_amplitude']:.6f}")

# Create animation
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Save artifacts after animation completes
fig.canvas.mpl_connect('close_event', lambda event: save_artifacts())

plt.show()
