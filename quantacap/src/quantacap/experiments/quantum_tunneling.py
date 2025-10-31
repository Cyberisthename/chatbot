"""1D quantum tunneling through a potential barrier."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from quantacap.utils.optional_import import optional_import


def _np():
    return optional_import("numpy", pip_name="numpy", purpose="quantum tunneling simulation")


@dataclass
class TunnelingResult:
    """Results from quantum tunneling simulation."""
    
    transmission: List[float]
    reflection: List[float]
    total_probability: List[float]
    final_transmission: float
    final_reflection: float
    parameters: Dict[str, float]
    steps: int


def simulate_tunneling(
    *,
    n: int = 1024,
    barrier_center: int = 512,
    barrier_width: int = 128,
    barrier_height: float = 5.0,
    energy: float = 2.0,
    steps: int = 2000,
    dt: float = 0.002,
    seed: int = 424242,
) -> TunnelingResult:
    """Simulate 1D quantum particle tunneling through a rectangular barrier.
    
    Uses the split-operator Fourier method to evolve the Schrödinger equation:
    iℏ ∂ψ/∂t = -(ℏ²/2m) ∂²ψ/∂x² + V(x)ψ
    
    Parameters
    ----------
    n : int
        Number of spatial grid points (should be power of 2 for FFT efficiency)
    barrier_center : int
        Center position of the barrier on the grid
    barrier_width : int
        Width of the rectangular barrier in grid points
    barrier_height : float
        Height of the potential barrier (in energy units)
    energy : float
        Initial kinetic energy of the wave packet
    steps : int
        Number of time steps to evolve
    dt : float
        Time step size
    seed : int
        Random seed (for potential future stochastic extensions)
        
    Returns
    -------
    TunnelingResult
        Contains transmission, reflection, and probability conservation data
    """
    if n <= 0 or steps <= 0:
        raise ValueError("n and steps must be positive")
    if barrier_width <= 0 or barrier_width >= n:
        raise ValueError("barrier_width must be positive and less than n")
    if barrier_center < barrier_width // 2 or barrier_center > n - barrier_width // 2:
        raise ValueError("barrier_center must be inside the grid with room for barrier_width")
    if energy <= 0:
        raise ValueError("energy must be positive")
    
    np = _np()
    
    # Set up spatial grid (units: ℏ=1, m=1/2)
    L = 50.0  # box size
    x = np.linspace(-L/2, L/2, n)
    dx = x[1] - x[0]
    
    # Set up momentum grid for FFT
    dk = 2.0 * np.pi / (n * dx)
    k = np.fft.fftfreq(n, dx) * 2.0 * np.pi
    
    # Create potential barrier (rectangular)
    V = np.zeros(n)
    barrier_start = barrier_center - barrier_width // 2
    barrier_end = barrier_center + barrier_width // 2
    V[barrier_start:barrier_end] = barrier_height
    
    # Initialize Gaussian wave packet with momentum k0 = sqrt(2*m*E) = sqrt(2*0.5*E) = sqrt(E)
    # Position it to the left of the barrier
    x0 = -L / 4.0  # initial position
    k0 = np.sqrt(energy)  # initial momentum
    sigma = 2.0  # wave packet width
    
    psi = np.exp(1j * k0 * x) * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    psi = psi.astype(np.complex128)
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi = psi / norm
    
    # Precompute evolution operators (split-operator method)
    # Kinetic energy operator in momentum space: T = ℏ²k²/(2m) = k²
    T = k ** 2
    exp_T = np.exp(-1j * T * dt)
    exp_V_half = np.exp(-1j * V * dt / 2.0)
    
    # Storage for observables
    transmission_history = []
    reflection_history = []
    total_prob_history = []
    
    # Time evolution
    for _ in range(steps):
        # Split-operator method: exp(-iH dt) ≈ exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2)
        
        # Step 1: Apply half potential
        psi = exp_V_half * psi
        
        # Step 2: Transform to momentum space, apply kinetic, transform back
        psi_k = np.fft.fft(psi)
        psi_k = exp_T * psi_k
        psi = np.fft.ifft(psi_k)
        
        # Step 3: Apply half potential again
        psi = exp_V_half * psi
        
        # Renormalize to prevent numerical drift
        norm_factor = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        if norm_factor > 0:
            psi = psi / norm_factor
        
        # Compute observables
        prob_density = np.abs(psi) ** 2
        
        # Transmission: probability to the right of the barrier
        trans = float(np.sum(prob_density[barrier_end:]) * dx)
        
        # Reflection: probability to the left of the barrier
        refl = float(np.sum(prob_density[:barrier_start]) * dx)
        
        # Total probability
        total = float(np.sum(prob_density) * dx)
        
        transmission_history.append(trans)
        reflection_history.append(refl)
        total_prob_history.append(total)
    
    return TunnelingResult(
        transmission=transmission_history,
        reflection=reflection_history,
        total_probability=total_prob_history,
        final_transmission=transmission_history[-1] if transmission_history else 0.0,
        final_reflection=reflection_history[-1] if reflection_history else 0.0,
        parameters={
            "n": float(n),
            "barrier_center": float(barrier_center),
            "barrier_width": float(barrier_width),
            "barrier_height": float(barrier_height),
            "energy": float(energy),
            "dt": float(dt),
            "steps": float(steps),
            "seed": float(seed),
        },
        steps=steps,
    )


def output_artifacts(
    result: TunnelingResult,
    *,
    out: str = "artifacts/tunneling_result.json",
    plot: str | None = "artifacts/tunneling_evolution.png",
) -> Dict[str, Any]:
    """Save tunneling simulation results to JSON and optionally plot.
    
    Parameters
    ----------
    result : TunnelingResult
        The simulation result to save
    out : str
        Path to JSON output file
    plot : str | None
        Path to PNG plot file, or None to skip plotting
        
    Returns
    -------
    dict
        Paths to created artifacts
    """
    # Prepare JSON payload
    payload = {
        "experiment": "quantum_tunneling",
        "params": result.parameters,
        "results": {
            "final_transmission": result.final_transmission,
            "final_reflection": result.final_reflection,
            "steps": result.steps,
            "transmission_history": result.transmission,
            "reflection_history": result.reflection,
            "total_probability_history": result.total_probability,
        },
    }
    
    # Write JSON
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    artifacts = {"json": str(out_path)}
    
    # Generate plot if requested
    if plot:
        plot_path = _maybe_plot_tunneling(result, plot)
        if plot_path:
            artifacts["plot"] = str(plot_path)
    
    return artifacts


def _maybe_plot_tunneling(result: TunnelingResult, out_path: str) -> Path | None:
    """Generate visualization of tunneling evolution."""
    try:
        mpl = optional_import(
            "matplotlib",
            pip_name="matplotlib",
            purpose="visualize tunneling evolution",
        )
        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except RuntimeError:
        return None
    
    np = _np()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top plot: Time evolution of transmission and reflection
    times = np.arange(len(result.transmission))
    ax1.plot(times, result.transmission, label="Transmission", color="tab:blue", linewidth=2)
    ax1.plot(times, result.reflection, label="Reflection", color="tab:red", linewidth=2)
    ax1.plot(times, result.total_probability, label="Total", color="black", 
             linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Probability")
    ax1.set_title("Quantum Tunneling: Probability Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Bottom plot: Transmission ratio vs time
    ax2.plot(times, result.transmission, color="tab:green", linewidth=2)
    ax2.axhline(y=result.final_transmission, color="tab:orange", linestyle="--", 
                label=f"Final: {result.final_transmission:.4f}")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Transmission probability")
    ax2.set_title(
        f"Tunneling Ratio (E={result.parameters['energy']:.1f}, "
        f"V={result.parameters['barrier_height']:.1f})"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, max(0.5, result.final_transmission * 1.2))
    
    plt.tight_layout()
    
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return out


def run_quantum_tunneling(
    *,
    energy: float = 2.0,
    barrier: float = 5.0,
    steps: int = 2000,
    n: int = 1024,
    barrier_width: int = 128,
    dt: float = 0.002,
    seed: int = 424242,
    out: str = "artifacts/tunneling_result.json",
    plot: str | None = "artifacts/tunneling_evolution.png",
) -> Dict[str, Any]:
    """High-level interface for quantum tunneling experiment.
    
    Parameters
    ----------
    energy : float
        Initial kinetic energy of the wave packet
    barrier : float
        Height of the potential barrier
    steps : int
        Number of time evolution steps
    n : int
        Number of spatial grid points
    barrier_width : int
        Width of the barrier in grid points
    dt : float
        Time step size
    seed : int
        Random seed
    out : str
        Path to JSON output
    plot : str | None
        Path to PNG plot, or None to skip
        
    Returns
    -------
    dict
        Summary with paths to artifacts
    """
    barrier_center = n // 2
    
    result = simulate_tunneling(
        n=n,
        barrier_center=barrier_center,
        barrier_width=barrier_width,
        barrier_height=barrier,
        energy=energy,
        steps=steps,
        dt=dt,
        seed=seed,
    )
    
    artifacts = output_artifacts(result, out=out, plot=plot)
    
    return {
        "experiment": "quantum_tunneling",
        "final_transmission": result.final_transmission,
        "final_reflection": result.final_reflection,
        "artifacts": artifacts,
    }
