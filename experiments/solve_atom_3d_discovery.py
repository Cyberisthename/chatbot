"""
3D Schrödinger Atom Solver - Discovery Mode
============================================

This is a physics-only, no-guessing, progressive-resolution 3D atom solver.

Goals:
  1. Solve ground state (and optionally excited states) from Schrödinger equation ONLY
  2. No precomputed orbitals, STOs, GTOs, or analytic solutions
  3. Produce real 3D density volumes (numpy arrays)
  4. Save ALL intermediate steps (64³ → 128³ → 256³)
  5. Generate 4K-ready renders
  6. Export everything for later analysis

FACTS-ONLY MODE: We do NOT:
  - Load or hardcode hydrogen orbitals
  - Smooth with gaussian filters
  - Symmetrize the density artificially
  - Enforce spherical symmetry

We ONLY use: Laplacian + potential + imaginary-time propagation + normalization
"""

import json
import math
import os
from pathlib import Path

import numpy as np

# Plotting imports inside functions for graceful degradation
VIZ = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    VIZ = False

# Optional imageio for GIF export
IMAGEIO_AVAILABLE = False
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    pass

# Optional scipy for better upsampling
SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "N_stages": [64, 128, 256],     # progressive refinements
    "box": 12.0,                    # domain: [-6, 6]³ in atomic units
    "Z": 1.0,                       # nuclear charge (1 = hydrogen)
    "softening": 0.3,               # nuclear softening to avoid singularity
    "dt": 0.002,                    # imaginary time step
    "steps_per_stage": 400,         # iterations per resolution stage
    "save_every": 50,               # save energy every N steps
    "centers": [[0.0, 0.0, 0.0]],   # nuclear positions (can add more for H2, etc)
    "seed": 424242,                 # reproducibility
}

FACTS_ONLY = True  # No shortcuts, no symmetrization, no smoothing


# ============================================================
# POTENTIAL FIELD
# ============================================================

def potential_field(shape, box, centers, Z, softening):
    """
    Build a 3D potential field V(x,y,z) for given nuclear centers.
    
    V(r) = -Z / sqrt((r - center)² + softening²)
    
    Args:
        shape: (N, N, N) grid shape
        box: physical size (domain is [-box/2, box/2]³)
        centers: list of [x, y, z] positions for nuclei
        Z: nuclear charge
        softening: softening parameter to avoid singularity
    
    Returns:
        V: 3D numpy array of potential
    """
    N = shape[0]
    L = box
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    V = np.zeros(shape, dtype=np.float64)
    
    for center in centers:
        xc, yc, zc = center
        r_sq = (X - xc)**2 + (Y - yc)**2 + (Z_grid - zc)**2
        V -= Z / np.sqrt(r_sq + softening**2)
    
    return V


# ============================================================
# 3D LAPLACIAN
# ============================================================

def laplacian3d(psi, dx):
    """
    3D finite-difference Laplacian using 6-point stencil.
    
    ∇²ψ ≈ (ψ(i+1,j,k) + ψ(i-1,j,k) + ψ(i,j+1,k) + ψ(i,j-1,k) + ψ(i,j,k+1) + ψ(i,j,k-1) - 6ψ(i,j,k)) / dx²
    
    We use np.roll for periodic-like boundary handling (but the box is large enough that boundaries don't matter).
    
    Args:
        psi: 3D wavefunction array
        dx: grid spacing
    
    Returns:
        lap: 3D Laplacian
    """
    lap = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) +
        np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2) -
        6.0 * psi
    ) / (dx * dx)
    return lap


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(psi, dx):
    """
    Normalize wavefunction so that ∫|ψ|² dV = 1
    
    Args:
        psi: 3D wavefunction
        dx: grid spacing
    
    Returns:
        psi_normalized, norm_value
    """
    dV = dx**3
    norm = np.sqrt((np.abs(psi)**2).sum() * dV)
    if norm <= 0:
        return psi, 0.0
    psi_normalized = psi / norm
    return psi_normalized, norm


# ============================================================
# ENERGY COMPUTATION
# ============================================================

def compute_energy(psi, V, dx):
    """
    Compute total energy E = <ψ| -½∇² + V |ψ>
    
    Args:
        psi: 3D wavefunction
        V: 3D potential
        dx: grid spacing
    
    Returns:
        E: total energy (float)
    """
    dV = dx**3
    lap = laplacian3d(psi, dx)
    
    kinetic = -0.5 * (np.conj(psi) * lap).real.sum() * dV
    potential = (np.conj(psi) * V * psi).real.sum() * dV
    
    E = kinetic + potential
    return float(E)


# ============================================================
# UPSAMPLING
# ============================================================

def upsample_wavefunction(psi, new_shape):
    """
    Upsample wavefunction to higher resolution.
    
    Uses scipy.ndimage.zoom if available, otherwise simple trilinear interpolation.
    
    Args:
        psi: 3D wavefunction at current resolution
        new_shape: target shape (N_new, N_new, N_new)
    
    Returns:
        psi_upsampled: wavefunction at new resolution
    """
    old_shape = psi.shape
    zoom_factors = tuple(new_shape[i] / old_shape[i] for i in range(3))
    
    if SCIPY_AVAILABLE:
        psi_upsampled = zoom(psi.real, zoom_factors, order=1)
        if np.iscomplexobj(psi):
            psi_upsampled = psi_upsampled + 1j * zoom(psi.imag, zoom_factors, order=1)
    else:
        # Simple nearest-neighbor fallback
        indices = np.meshgrid(
            np.linspace(0, old_shape[0]-1, new_shape[0]).astype(int),
            np.linspace(0, old_shape[1]-1, new_shape[1]).astype(int),
            np.linspace(0, old_shape[2]-1, new_shape[2]).astype(int),
            indexing='ij'
        )
        psi_upsampled = psi[indices[0], indices[1], indices[2]]
    
    return psi_upsampled


# ============================================================
# IMAGINARY-TIME EVOLUTION (SINGLE STAGE)
# ============================================================

def evolve_stage(N, box, Z, centers, softening, dt, steps, psi_init=None, save_every=50):
    """
    Evolve wavefunction in imaginary time for one resolution stage.
    
    Args:
        N: grid size (N×N×N)
        box: physical box size
        Z: nuclear charge
        centers: list of nuclear positions
        softening: nuclear softening
        dt: imaginary time step (will be auto-scaled based on dx for stability)
        steps: number of evolution steps
        psi_init: initial wavefunction (if None, start from random)
        save_every: save energy every N steps
    
    Returns:
        dict with psi, density, V, energies, dx
    """
    # Build potential
    V = potential_field((N, N, N), box, centers, Z, softening)
    
    dx = box / N
    
    # Auto-scale dt for numerical stability
    # CFL condition for diffusion: dt ~ dx²
    # Scale dt proportionally to maintain stability at all resolutions
    dt_scaled = dt * (dx / (box / 64))**2
    print(f"  Using time step dt={dt_scaled:.6f} (scaled for dx={dx:.6f})")
    
    # Initialize wavefunction
    if psi_init is None:
        # Start from random gaussian-like blob
        np.random.seed(CONFIG["seed"])
        x = np.linspace(-box/2, box/2, N)
        y = np.linspace(-box/2, box/2, N)
        z = np.linspace(-box/2, box/2, N)
        X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
        
        r_sq = X**2 + Y**2 + Z_grid**2
        psi = np.exp(-r_sq / 4.0) * (1.0 + 0.05 * np.random.randn(*X.shape))
    else:
        psi = psi_init.copy()
    
    # Normalize
    psi, _ = normalize(psi, dx)
    
    energies = []
    prev_E = None
    
    # Imaginary-time evolution
    for step in range(steps):
        lap = laplacian3d(psi, dx)
        
        # dpsi/dτ = ½∇²ψ - Vψ
        dpsi = 0.5 * lap - V * psi
        psi = psi + dt_scaled * dpsi
        
        # Renormalize
        psi, _ = normalize(psi, dx)
        
        # Save energy periodically
        if step % save_every == 0 or step == steps - 1:
            E = compute_energy(psi, V, dx)
            energies.append({"step": step, "E": E})
            print(f"  N={N}, step={step}/{steps}, E={E:.6f}")
            
            # Check for divergence
            if prev_E is not None and abs(E) > 10 * abs(prev_E) and abs(E) > 1.0:
                print(f"  ⚠️  Warning: Energy diverging! Stopping evolution.")
                print(f"  Consider reducing time step or increasing softening.")
                break
            prev_E = E
    
    density = np.abs(psi)**2
    
    return {
        "psi": psi,
        "density": density,
        "V": V,
        "energies": energies,
        "dx": dx,
        "N": N,
        "dt_used": dt_scaled,
    }


# ============================================================
# PROGRESSIVE MULTI-STAGE SOLVER
# ============================================================

def run_all_stages():
    """
    Run progressive-resolution solver: 64³ → 128³ → 256³
    
    Each stage:
      1. Build/recompute potential at current resolution
      2. Evolve wavefunction in imaginary time
      3. Save all artifacts
      4. Upsample to next resolution
    
    Handles MemoryError gracefully.
    """
    print("="*60)
    print("3D SCHRÖDINGER ATOM SOLVER - DISCOVERY MODE")
    print("="*60)
    print(f"Configuration:")
    print(f"  Stages: {CONFIG['N_stages']}")
    print(f"  Box size: {CONFIG['box']} a.u.")
    print(f"  Nuclear charge Z: {CONFIG['Z']}")
    print(f"  Centers: {CONFIG['centers']}")
    print(f"  Softening: {CONFIG['softening']}")
    print(f"  Time step dt: {CONFIG['dt']}")
    print(f"  Steps per stage: {CONFIG['steps_per_stage']}")
    print(f"  FACTS_ONLY mode: {FACTS_ONLY}")
    print("="*60)
    
    outdir = Path("artifacts/real_atom_3d")
    outdir.mkdir(parents=True, exist_ok=True)
    
    psi = None
    all_stage_results = []
    
    for stage_idx, N in enumerate(CONFIG["N_stages"]):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx+1}/{len(CONFIG['N_stages'])}: N={N}³")
        print(f"{'='*60}")
        
        try:
            # Upsample previous stage if available
            if psi is not None:
                print(f"Upsampling wavefunction from {psi.shape[0]}³ to {N}³...")
                psi = upsample_wavefunction(psi, (N, N, N))
                psi, _ = normalize(psi, CONFIG["box"] / N)
            
            # Evolve at this resolution
            result = evolve_stage(
                N=N,
                box=CONFIG["box"],
                Z=CONFIG["Z"],
                centers=CONFIG["centers"],
                softening=CONFIG["softening"],
                dt=CONFIG["dt"],
                steps=CONFIG["steps_per_stage"],
                psi_init=psi,
                save_every=CONFIG["save_every"],
            )
            
            psi = result["psi"]
            
            # Save artifacts for this stage
            save_stage_artifacts(result, outdir, stage_idx, N)
            
            all_stage_results.append({
                "stage": stage_idx,
                "N": N,
                "final_energy": result["energies"][-1]["E"],
                "energies": result["energies"],
            })
            
            print(f"✅ Stage {stage_idx+1} complete: N={N}³, E={result['energies'][-1]['E']:.6f}")
            
        except MemoryError as e:
            print(f"⚠️  MemoryError at N={N}³: {e}")
            print(f"Stopping at previous stage (N={CONFIG['N_stages'][stage_idx-1] if stage_idx > 0 else 'none'}³)")
            break
        except Exception as e:
            print(f"❌ Error at stage N={N}³: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Save master descriptor
    save_master_descriptor(all_stage_results, outdir)
    
    # Generate final renders
    print(f"\n{'='*60}")
    print("GENERATING FINAL RENDERS...")
    print(f"{'='*60}")
    
    if len(all_stage_results) > 0:
        # Use highest-resolution stage for final renders
        highest_N = all_stage_results[-1]["N"]
        density = np.load(outdir / f"density_N{highest_N}.npy")
        generate_renders(density, outdir, highest_N)
    
    print(f"\n{'='*60}")
    print("✅ ALL STAGES COMPLETE")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to: {outdir}")
    print(f"\nTo view results:")
    print(f"  {outdir}/atom_mip_xy.png")
    print(f"  {outdir}/atom_mip_xz.png")
    print(f"  {outdir}/atom_mip_yz.png")
    if IMAGEIO_AVAILABLE:
        print(f"  {outdir}/atom_spin.gif")
    print(f"  {outdir}/atom3d_descriptor.json")
    print("\n" + "="*60)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_stage_artifacts(result, outdir, stage_idx, N):
    """Save all artifacts for a single stage."""
    print(f"Saving artifacts for stage N={N}³...")
    
    # Save arrays
    np.save(outdir / f"psi_N{N}.npy", result["psi"])
    np.save(outdir / f"density_N{N}.npy", result["density"])
    np.save(outdir / f"potential_N{N}.npy", result["V"])
    
    # Save energy history
    with open(outdir / f"energy_N{N}.json", "w") as f:
        json.dump(result["energies"], f, indent=2)
    
    print(f"  Saved: psi_N{N}.npy, density_N{N}.npy, potential_N{N}.npy, energy_N{N}.json")


def save_master_descriptor(all_stage_results, outdir):
    """Save master JSON descriptor for all stages."""
    descriptor = {
        "name": "REAL-ATOM-3D-DISCOVERY-V1",
        "note": "derived from Schrödinger equation in imaginary time; no precomputed orbitals",
        "facts_only": FACTS_ONLY,
        "config": CONFIG,
        "stages": all_stage_results,
        "artifacts": {
            "densities": [f"artifacts/real_atom_3d/density_N{r['N']}.npy" for r in all_stage_results],
            "wavefunctions": [f"artifacts/real_atom_3d/psi_N{r['N']}.npy" for r in all_stage_results],
            "potentials": [f"artifacts/real_atom_3d/potential_N{r['N']}.npy" for r in all_stage_results],
            "energies": [f"artifacts/real_atom_3d/energy_N{r['N']}.json" for r in all_stage_results],
            "renders": {
                "mip_xy": "artifacts/real_atom_3d/atom_mip_xy.png",
                "mip_xz": "artifacts/real_atom_3d/atom_mip_xz.png",
                "mip_yz": "artifacts/real_atom_3d/atom_mip_yz.png",
                "spin_gif": "artifacts/real_atom_3d/atom_spin.gif",
            },
        },
    }
    
    with open(outdir / "atom3d_descriptor.json", "w") as f:
        json.dump(descriptor, f, indent=2)
    
    print(f"✅ Master descriptor saved: {outdir}/atom3d_descriptor.json")


# ============================================================
# RENDERING
# ============================================================

def render_volume_maxip(volume, outpath, dpi=1000, title="Atom Density", cmap="inferno"):
    """
    Render a 2D max-intensity projection from 3D volume.
    
    Args:
        volume: 3D numpy array
        outpath: output file path
        dpi: resolution (higher = better quality)
        title: plot title
        cmap: colormap
    """
    if not VIZ:
        print("⚠️  matplotlib not available, skipping render")
        return
    
    # Normalize to [0, 1]
    img = volume / (volume.max() + 1e-12)
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    im = ax.imshow(img, origin='lower', cmap=cmap, interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  Rendered: {outpath}")


def generate_renders(density, outdir, N):
    """Generate all renders: 3 orthogonal MIPs and optional spin GIF."""
    print("Generating renders...")
    
    # Max-intensity projections along each axis
    mip_xy = density.max(axis=2)  # max along z → view from top
    mip_xz = density.max(axis=1)  # max along y → view from side
    mip_yz = density.max(axis=0)  # max along x → view from other side
    
    render_volume_maxip(mip_xy, outdir / "atom_mip_xy.png", dpi=1000, title="Atom Density (XY view)", cmap="inferno")
    render_volume_maxip(mip_xz, outdir / "atom_mip_xz.png", dpi=1000, title="Atom Density (XZ view)", cmap="inferno")
    render_volume_maxip(mip_yz, outdir / "atom_mip_yz.png", dpi=1000, title="Atom Density (YZ view)", cmap="inferno")
    
    # Optional: 360° spin GIF
    if IMAGEIO_AVAILABLE and VIZ:
        generate_spin_gif(density, outdir)
    else:
        print("⚠️  imageio not available, skipping spin GIF")


def generate_spin_gif(density, outdir):
    """Generate a 360° spin animation of the atom."""
    print("Generating spin GIF...")
    
    frames = []
    n_frames = 36
    
    # Create rotation by viewing from different angles
    # For simplicity, we'll rotate the MIP view
    mip = density.max(axis=2)
    
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        # Rotate by creating a weighted blend (simple effect)
        # For true 3D rotation we'd need more complex code, but this gives a spin effect
        mip_rotated = np.roll(mip, int(5 * np.sin(angle)), axis=0)
        mip_rotated = np.roll(mip_rotated, int(5 * np.cos(angle)), axis=1)
        
        # Normalize and convert to uint8
        img = (mip_rotated / (mip_rotated.max() + 1e-12) * 255).astype(np.uint8)
        frames.append(img)
    
    imageio.mimsave(outdir / "atom_spin.gif", frames, duration=0.1)
    print(f"  Rendered: {outdir}/atom_spin.gif")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  3D SCHRÖDINGER ATOM SOLVER")
    print("  DISCOVERY MODE - NO PRECOMPUTED ORBITALS")
    print("  Physics only: ∇²ψ + V(r)ψ = Eψ")
    print("="*60 + "\n")
    
    run_all_stages()
    
    print("\n" + "="*60)
    print("DONE! View results:")
    print("  python -m experiments.solve_atom_3d_discovery")
    print("\nThen open:")
    print("  artifacts/real_atom_3d/atom_mip_xy.png")
    print("  artifacts/real_atom_3d/atom_mip_xz.png")
    print("  artifacts/real_atom_3d/atom_mip_yz.png")
    print("  artifacts/real_atom_3d/atom_spin.gif  (if created)")
    print("  artifacts/real_atom_3d/atom3d_descriptor.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
