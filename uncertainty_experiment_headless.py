#!/usr/bin/env python3
"""
Quantum-Style Uncertainty Collapse Experiment (Headless Version)

Shows how random noise (chaos) gradually collapses into a stable quantum-like
amplitude distribution — an easy visual "quantum stabilization" demo.

This headless version runs without displaying the animation and generates
both artifacts and a static image file.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

# Store data for plotting
frame_data = []
amplitude_history = []

print(f"🔬 Running Quantum Uncertainty Collapse Experiment...")
print(f"   Points: {n_points}, Frames: {frames}, Decay: {decay}")

# Run simulation
for frame in range(frames):
    phase = np.exp(1j * (frame / 10))
    psi = np.exp(-x**2) * phase + amplitude * noise[frame] * 0.1
    amplitude *= decay
    prob = np.abs(psi)**2
    
    # Store every 20th frame for visualization
    if frame % 20 == 0:
        frame_data.append((frame, prob / np.max(prob)))
    
    amplitude_history.append(amplitude)
    
    if frame % 50 == 0:
        print(f"   Frame {frame:3d} | randomness={amplitude:.6f}")

print(f"   Frame {frames:3d} | randomness={amplitude:.6f}")

# Generate visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot selected frames
for frame_idx, prob in frame_data:
    alpha = 0.3 + 0.7 * (frame_idx / frames)
    color = plt.cm.viridis(frame_idx / frames)
    ax1.plot(x, prob, color=color, alpha=alpha, label=f'Frame {frame_idx}' if frame_idx % 60 == 0 else '')

ax1.set_title("Quantum Uncertainty Collapse - Probability Distribution Evolution", fontsize=12, fontweight='bold')
ax1.set_xlabel("Position")
ax1.set_ylabel("Normalized Amplitude²")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Plot amplitude decay
ax2.plot(amplitude_history, color='cyan', linewidth=2)
ax2.set_title("Noise Amplitude Decay (Decoherence Process)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Frame")
ax2.set_ylabel("Amplitude")
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
artifacts_dir = Path(__file__).parent / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
figure_path = artifacts_dir / "uncertainty_experiment.png"
plt.savefig(figure_path, dpi=150, bbox_inches='tight')
print(f"\n📊 Visualization saved to {figure_path}")

# Calculate final statistics
prob_final = np.abs(psi)**2
prob_normalized = prob_final / np.sum(prob_final)

results = {
    "experiment": "quantum_uncertainty_collapse",
    "version": "1.0",
    "parameters": {
        "n_points": n_points,
        "frames": frames,
        "decay_rate": decay,
        "initial_amplitude": 1.0
    },
    "results": {
        "final_amplitude": float(amplitude),
        "final_entropy": float(np.var(prob_final)),
        "final_mean": float(np.mean(prob_final)),
        "final_std": float(np.std(prob_final)),
        "position_mean": float(np.sum(x * prob_normalized)),
        "position_std": float(np.sqrt(np.sum((x - np.sum(x * prob_normalized))**2 * prob_normalized))),
        "max_amplitude": float(np.max(prob_final)),
        "min_amplitude": float(np.min(prob_final))
    },
    "physics": {
        "interpretation": "Digital analog of quantum decoherence",
        "process": "Chaos (quantum randomness) stabilizing into classical order",
        "convergence": "Amplitude² converges to Gaussian distribution",
        "decoherence_rate": f"{(1-decay)*100:.1f}% per frame"
    },
    "notes": "Simulates decoherence: chaos (quantum randomness) stabilizing into classical order"
}

# Save artifacts
output_path = artifacts_dir / "uncertainty_experiment.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Artifacts saved to {output_path}")
print(f"\n📈 Results:")
print(f"   Final entropy (variance): {results['results']['final_entropy']:.6f}")
print(f"   Final amplitude: {results['results']['final_amplitude']:.6f}")
print(f"   Position mean: {results['results']['position_mean']:.6f}")
print(f"   Position std: {results['results']['position_std']:.6f}")
print(f"   Decoherence rate: {results['physics']['decoherence_rate']}")
print(f"\n🎯 Experiment complete!")
