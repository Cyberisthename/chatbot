#!/usr/bin/env python3
"""
REAL ATOM 3D DISCOVERY — FULL 4K VERSION (V2)
==============================================

Simulate and render the first-ever fact-based, 3D quantum atom image using
nothing but physical constants — no precomputed orbitals or analytic hydrogenic solutions.

PHYSICS MODEL (NO ASSUMPTIONS)
------------------------------
- Schrödinger equation in imaginary time:  dψ/dτ = (1/2)∇²ψ - V(r)ψ
- Potential: V(r) = -Z / sqrt(r² + ε²)
- Boundary: ψ → 0 at |r| = L/2
- Method: Imaginary-time propagation via split-operator FFT (Strang splitting)
- Units: atomic units (ħ = mₑ = e = 1)
- Random normalized ψ₀(x,y,z)
- Normalize after every step so that ∫|ψ|² dV = 1
- Update rule:
    ψ ← FFT⁻¹[exp(-½k²dt) FFT(ψ)]
    ψ ← exp(-V dt) ψ
    ψ ← FFT⁻¹[exp(-½k²dt) FFT(ψ)]
- Absorbing boundary mask in outer 10% radius: ψ *= cos⁸(πr/Rmax)
- Adaptive dt: halve dt if total energy increases (with retry logic)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
VIZ_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VIZ_AVAILABLE = True
except Exception:
    pass

VIDEO_AVAILABLE = False
try:
    import imageio
    VIDEO_AVAILABLE = True
except Exception:
    pass

ISOSURFACE_AVAILABLE = False
try:
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ISOSURFACE_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, float | int | bool] = {
    "Z": 1.0,
    "L": 20.0,
    "ε": 0.15,
    "N": 512,
    "steps": 2000,
    "dt": 0.001,
    "save_every": 100,
    "seed": 424242,
    "facts_only": True,
}

MIN_DT_FACTOR = 1e-4  # prevent dt from collapsing to zero
ENERGY_TOL = 1e-8     # tolerance for energy increase detection


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def print_hardware_info() -> None:
    """Print basic hardware / environment information."""
    print("\n" + "=" * 70)
    print("HARDWARE & ENVIRONMENT INFO")
    print("=" * 70)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or 'Unknown'}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(
            "RAM: "
            f"{mem.total / (1024 ** 3):.2f} GB "
            f"(available: {mem.available / (1024 ** 3):.2f} GB)"
        )
    except Exception:
        print("RAM: psutil not available")

    print("=" * 70 + "\n")


def print_memory_usage(label: str = "") -> None:
    """Print current RSS memory usage if psutil is installed."""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Memory usage {label}: {mem_info.rss / (1024 ** 3):.3f} GB")
    except Exception:
        pass


def estimate_memory_gb(N: int) -> float:
    """Estimate memory consumption in GB for key arrays."""
    # Arrays: psi (complex128), density (float64), potential (float64),
    # k_sq (float64), absorbing mask (float64) ≈ 5 arrays.
    bytes_per_point = 16 + 8 + 8 + 8 + 8  # ≈ 48 bytes
    total_bytes = N ** 3 * bytes_per_point
    return total_bytes / (1024 ** 3)


def auto_adjust_grid_for_memory(config: Dict[str, float | int | bool]) -> None:
    """Reduce N if current memory budget is insufficient."""
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        estimated_gb = estimate_memory_gb(config["N"])
        print(f"Estimated memory needed: ~{estimated_gb:.2f} GB")

        # If requirement exceeds 85% of free RAM, drop to 256³ (spec requirement)
        if estimated_gb > 0.85 * available_gb and config["N"] > 256:
            print("\n⚠️  Detected limited RAM. Automatically reducing grid to N=256³.")
            config["N"] = 256
            estimated_gb = estimate_memory_gb(config["N"])
            print(f"New estimate for N=256: ~{estimated_gb:.2f} GB")
    except Exception:
        # psutil might be unavailable; continue silently
        pass


# ---------------------------------------------------------------------------
# Physics engine components
# ---------------------------------------------------------------------------
def build_potential(N: int, L: float, Z: float, epsilon: float) -> np.ndarray:
    """Compute softened Coulomb potential on the grid."""
    axis = np.linspace(-L / 2, L / 2, N)
    X, Y, Z_grid = np.meshgrid(axis, axis, axis, indexing="ij")
    r_sq = X ** 2 + Y ** 2 + Z_grid ** 2
    V = -Z / np.sqrt(r_sq + epsilon ** 2)
    return V.astype(np.float64)


def build_k_squared(N: int, L: float) -> np.ndarray:
    """Precompute |k|² for FFT operations."""
    k = np.fft.fftfreq(N, d=L / N) * 2.0 * math.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k_sq = kx ** 2 + ky ** 2 + kz ** 2
    return k_sq.astype(np.float64)


def build_absorbing_mask(N: int, L: float) -> np.ndarray:
    """Construct cos⁸ absorbing mask in the outer 10% of the radius."""
    axis = np.linspace(-L / 2, L / 2, N)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R_max = L / 2.0
    R_absorb = 0.9 * R_max

    mask = np.ones_like(r, dtype=np.float64)
    absorb_region = r > R_absorb
    if np.any(absorb_region):
        scaled = (r[absorb_region] - R_absorb) / (R_max - R_absorb)
        mask[absorb_region] = np.cos(0.5 * math.pi * scaled) ** 8
    return mask


def initialize_wavefunction(N: int, L: float, seed: int) -> np.ndarray:
    """Create a random, roughly spherical initial wavefunction."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(-L / 2, L / 2, N)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r_sq = X ** 2 + Y ** 2 + Z ** 2

    sigma = 2.0
    psi = np.exp(-r_sq / (2.0 * sigma ** 2))
    psi *= 1.0 + 0.05 * rng.standard_normal(size=psi.shape)
    return psi.astype(np.complex128)


def normalize_wavefunction(psi: np.ndarray, dV: float) -> Tuple[np.ndarray, float]:
    """Normalize ψ so that ∫|ψ|² dV = 1."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2).real * dV)
    if norm > 0:
        psi /= norm
    return psi, float(norm)


def compute_energy(
    psi: np.ndarray,
    V: np.ndarray,
    k_sq: np.ndarray,
    dV: float,
) -> Tuple[float, float, float]:
    """Compute total, kinetic, and potential energy."""
    psi_k = np.fft.fftn(psi)
    laplacian_psi = np.fft.ifftn(-k_sq * psi_k)

    kinetic = (-0.5 * np.sum(np.conj(psi) * laplacian_psi).real) * dV
    potential = np.sum((np.abs(psi) ** 2) * V).real * dV

    total = kinetic + potential
    return float(total), float(kinetic), float(potential)


def compute_expectation_r(psi: np.ndarray, L: float, dV: float) -> Tuple[float, float]:
    """Compute ⟨r⟩ and ⟨r²⟩."""
    N = psi.shape[0]
    axis = np.linspace(-L / 2, L / 2, N)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    density = np.abs(psi) ** 2

    r_mean = np.sum(density * r).real * dV
    r2_mean = np.sum(density * (r ** 2)).real * dV
    return float(r_mean), float(r2_mean)


def split_operator_step(
    psi: np.ndarray,
    V: np.ndarray,
    k_sq: np.ndarray,
    dt: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Perform a single Strang split-operator step in imaginary time."""
    kinetic_factor = np.exp(-0.5 * k_sq * dt)

    psi_k = np.fft.fftn(psi)
    psi_k *= kinetic_factor
    psi = np.fft.ifftn(psi_k)

    psi *= np.exp(-V * dt)

    psi_k = np.fft.fftn(psi)
    psi_k *= kinetic_factor
    psi = np.fft.ifftn(psi_k)

    psi *= mask
    return psi.astype(np.complex128)


def compute_radial_profile(
    density: np.ndarray,
    L: float,
    dV: float,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial density ρ(r) and radial probability P(r)."""
    N = density.shape[0]
    axis = np.linspace(-L / 2, L / 2, N)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2).ravel()
    prob_density = density.ravel() * dV  # probability per voxel

    r_max = L / 2.0
    r_bins = np.linspace(0.0, r_max, n_bins + 1)
    shell_prob, _ = np.histogram(r, bins=r_bins, weights=prob_density)
    shell_counts, _ = np.histogram(r, bins=r_bins)

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    dr = np.diff(r_bins)
    shell_volumes = (4.0 / 3.0) * math.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)

    rho = np.zeros_like(r_centers)
    P = np.zeros_like(r_centers)

    mask = shell_volumes > 0
    rho[mask] = shell_prob[mask] / shell_volumes[mask]
    P[mask] = shell_prob[mask] / dr[mask]

    return r_centers, rho, P


# ---------------------------------------------------------------------------
# Imaginary-time solver
# ---------------------------------------------------------------------------
def imaginary_time_propagation(config: Dict[str, float | int | bool]) -> Dict[str, object]:
    """Run the split-operator imaginary-time evolution."""
    Z = float(config["Z"])
    L = float(config["L"])
    epsilon = float(config["ε"])
    N = int(config["N"])
    steps = int(config["steps"])
    dt_initial = float(config["dt"])
    save_every = int(config["save_every"])
    seed = int(config["seed"])

    print("\n" + "=" * 70)
    print("INITIALIZING QUANTUM SOLVER")
    print("=" * 70)
    print(f"Nuclear charge Z: {Z}")
    print(f"Box size L: {L} a.u.")
    print(f"Grid size N: {N}³")
    print(f"Softening ε: {epsilon}")
    print(f"Time step dt (initial): {dt_initial}")
    print(f"Total steps: {steps}")
    print(f"Seed: {seed}")
    print("=" * 70 + "\n")

    dx = L / N
    dV = dx ** 3
    print(f"Grid spacing dx: {dx:.6f} a.u.")
    print(f"Voxel volume dV: {dV:.6e} a.u.³")

    print("\nBuilding operators...")
    V = build_potential(N, L, Z, epsilon)
    k_sq = build_k_squared(N, L)
    mask = build_absorbing_mask(N, L)
    print("  ✓ Potential field")
    print("  ✓ k² operator")
    print("  ✓ Absorbing mask")

    print_memory_usage("after operator construction")

    print("\nInitializing wavefunction...")
    psi = initialize_wavefunction(N, L, seed)
    psi, norm = normalize_wavefunction(psi, dV)
    print("  ✓ Wavefunction initialized (norm = 1.0)")

    energy_history = []
    E, E_kin, E_pot = compute_energy(psi, V, k_sq, dV)
    r_mean, r2_mean = compute_expectation_r(psi, L, dV)
    print(
        f"\nInitial energy: E = {E:.6f} Ha (K = {E_kin:.6f}, U = {E_pot:.6f})"
    )

    dt_current = dt_initial
    dt_min = max(dt_initial * MIN_DT_FACTOR, 1e-8)
    step = 0
    start = time.time()

    latest_norm = norm
    latest_r_mean = r_mean
    latest_r2_mean = r2_mean

    print("\n" + "=" * 70)
    print("IMAGINARY-TIME EVOLUTION")
    print("=" * 70)

    while step < steps:
        psi_trial = split_operator_step(psi, V, k_sq, dt_current, mask)
        psi_trial, norm = normalize_wavefunction(psi_trial, dV)

        E_trial, E_kin_trial, E_pot_trial = compute_energy(psi_trial, V, k_sq, dV)

        if E_trial > E + ENERGY_TOL:
            dt_current *= 0.5
            if dt_current < dt_min:
                print(
                    "  ⚠️  dt reached minimum threshold; accepting step despite"
                    f" energy increase (E = {E_trial:.6f})"
                )
            else:
                print(
                    f"  ⚠️  Energy increased ({E_trial:.6f} > {E:.6f}); "
                    f"reducing dt to {dt_current:.6g} and retrying step"
                )
                continue

        psi = psi_trial
        E = E_trial
        E_kin = E_kin_trial
        E_pot = E_pot_trial
        latest_norm = norm

        step += 1

        if step % save_every == 0 or step == steps:
            latest_r_mean, latest_r2_mean = compute_expectation_r(psi, L, dV)
            energy_history.append(
                {
                    "step": step,
                    "E": E,
                    "E_kin": E_kin,
                    "E_pot": E_pot,
                    "norm": latest_norm,
                    "r_mean": latest_r_mean,
                    "r2_mean": latest_r2_mean,
                    "dt": dt_current,
                }
            )

            elapsed = time.time() - start
            rate = step / elapsed if elapsed > 0 else 0.0
            print(
                f"Step {step:5d}/{steps} | E = {E:+.6f} Ha | "
                f"K = {E_kin:.4f} | U = {E_pot:.4f} | "
                f"<r> = {latest_r_mean:.4f} | dt = {dt_current:.2e} | "
                f"{rate:.1f} steps/s"
            )

    elapsed_total = time.time() - start
    print(f"\n✅ Evolution complete in {elapsed_total:.1f} seconds")
    print(f"Final energy: E = {E:.6f} Ha")

    print("\nComputing final observables...")
    density = np.abs(psi) ** 2
    radial_r, radial_rho, radial_P = compute_radial_profile(density, L, dV)
    analytic_rho = (1.0 / math.pi) * np.exp(-2.0 * radial_r)
    chi_squared = float(np.mean((radial_rho - analytic_rho) ** 2))
    print(f"  χ² deviation from analytic 1s: {chi_squared:.6e}")

    return {
        "psi": psi,
        "density": density,
        "V": V,
        "energy_history": energy_history,
        "radial_r": radial_r,
        "radial_rho": radial_rho,
        "radial_P": radial_P,
        "analytic_rho": analytic_rho,
        "chi_squared": chi_squared,
        "config": config,
        "E_final": E,
        "r_mean": latest_r_mean,
        "r2_mean": latest_r2_mean,
        "norm_final": latest_norm,
        "dx": dx,
        "dV": dV,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artifacts(results: Dict[str, object], output_dir: str) -> None:
    """Persist arrays, CSV logs, and summary report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    np.save(out_path / "density.npy", results["density"], allow_pickle=False)
    np.save(out_path / "psi.npy", results["psi"], allow_pickle=False)
    np.save(out_path / "potential.npy", results["V"], allow_pickle=False)
    print("  ✓ density.npy")
    print("  ✓ psi.npy")
    print("  ✓ potential.npy")

    with (out_path / "energy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "E",
                "E_kin",
                "E_pot",
                "norm",
                "r_mean",
                "r2_mean",
                "dt",
            ],
        )
        writer.writeheader()
        writer.writerows(results["energy_history"])
    print("  ✓ energy.csv")

    with (out_path / "radial.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "rho", "P", "analytic_rho"])
        for r_val, rho_val, P_val, analytic in zip(
            results["radial_r"],
            results["radial_rho"],
            results["radial_P"],
            results["analytic_rho"],
        ):
            writer.writerow([r_val, rho_val, P_val, analytic])
    print("  ✓ radial.csv")

    report = {
        "name": "REAL-ATOM-3D-V2",
        "description": "Physics-only 3D quantum atom from Schrödinger equation",
        "Z": results["config"]["Z"],
        "box": results["config"]["L"],
        "softening": results["config"]["ε"],
        "N": results["config"]["N"],
        "steps": results["config"]["steps"],
        "dt_initial": results["config"]["dt"],
        "save_every": results["config"]["save_every"],
        "seed": results["config"]["seed"],
        "E_final": results["E_final"],
        "norm": results["norm_final"],
        "r_mean": results["r_mean"],
        "r2_mean": results["r2_mean"],
        "analytic_overlap": results["chi_squared"],
        "facts_only": results["config"]["facts_only"],
        "dx": results["dx"],
        "dV": results["dV"],
    }

    with (out_path / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print("  ✓ report.json")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------
def render_orthogonal_slices(
    density: np.ndarray,
    L: float,
    output_dir: str,
    dpi: int = 1000,
) -> None:
    if not VIZ_AVAILABLE:
        print("⚠️  matplotlib not available; skipping slice renders")
        return

    print("\nRendering orthogonal slices (4K)...")
    out_path = Path(output_dir)
    N = density.shape[0]
    mid = N // 2
    density_norm = density / (density.max() + 1e-12)
    extent = [-L / 2, L / 2, -L / 2, L / 2]

    slices = {
        "xy": density_norm[:, :, mid],
        "xz": density_norm[:, mid, :],
        "yz": density_norm[mid, :, :],
    }

    for name, data in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap="inferno",
            interpolation="bilinear",
        )
        ax.set_xlabel(f"{name[0]} (a.u.)")
        ax.set_ylabel(f"{name[1]} (a.u.)")
        ax.set_title(f"Atom density {name.upper()} slice")
        fig.colorbar(im, ax=ax, label="ρ (normalized)")
        plt.tight_layout()
        fig.savefig(out_path / f"atom_{name}_4k.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ atom_{name}_4k.png")


def render_isosurfaces(
    density: np.ndarray,
    L: float,
    output_dir: str,
    levels: Tuple[float, ...] = (0.6, 0.3, 0.1),
    dpi: int = 1000,
) -> None:
    if not (VIZ_AVAILABLE and ISOSURFACE_AVAILABLE):
        print("⚠️  Isosurface rendering unavailable (needs matplotlib + scikit-image)")
        return

    print("\nRendering isosurfaces (4K)...")
    out_path = Path(output_dir)
    density_norm = density / (density.max() + 1e-12)
    N = density.shape[0]

    for level in levels:
        try:
            verts, faces, _, _ = marching_cubes(density_norm, level=level)
            verts = verts / N * L - L / 2

            fig = plt.figure(figsize=(8, 8), dpi=dpi // 2)
            ax = fig.add_subplot(111, projection="3d")
            mesh = Poly3DCollection(verts[faces], alpha=0.35, edgecolor="none")
            mesh.set_facecolor((0.2, 0.9, 0.9, 0.8))
            ax.add_collection3d(mesh)

            ax.set_xlim(-L / 2, L / 2)
            ax.set_ylim(-L / 2, L / 2)
            ax.set_zlim(-L / 2, L / 2)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.set_xlabel("x (a.u.)")
            ax.set_ylabel("y (a.u.)")
            ax.set_zlabel("z (a.u.)")
            ax.set_title(f"Atom isosurface (ρ = {level:.1f})")
            plt.tight_layout()
            fig.savefig(
                out_path / f"isosurface_{level:.1f}.png",
                dpi=dpi // 2,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"  ✓ isosurface_{level:.1f}.png")
        except Exception as exc:  # pragma: no cover - visualization fallback
            print(f"  ⚠️  Failed to render isosurface at level {level}: {exc}")


def render_orbit_video(
    density: np.ndarray,
    L: float,
    output_dir: str,
    n_frames: int = 720,
    fps: int = 30,
    dpi: int = 480,
    iso_level: float = 0.3,
) -> None:
    if not (VIZ_AVAILABLE and VIDEO_AVAILABLE and ISOSURFACE_AVAILABLE):
        print("⚠️  Orbit video skipped (requires matplotlib, imageio, scikit-image)")
        return

    print(f"\nRendering orbit video ({n_frames} frames @ {fps} fps)...")
    out_path = Path(output_dir)

    density_norm = density / (density.max() + 1e-12)
    verts, faces, _, _ = marching_cubes(density_norm, level=iso_level)
    N = density.shape[0]
    verts = verts / N * L - L / 2

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.35, edgecolor="none")
    mesh.set_facecolor((0.2, 0.9, 0.9, 0.8))
    ax.add_collection3d(mesh)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    ax.set_zlim(-L / 2, L / 2)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.axis("off")
    ax.set_title("Atom isosurface (orbit)")

    output_path = out_path / "atom_orbit.mp4"
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=9,
        bitrate="16M",
    ) as writer:
        for frame in range(n_frames):
            angle = 360.0 * frame / n_frames
            ax.view_init(elev=25, azim=angle)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            if (frame + 1) % 120 == 0:
                print(f"  Frame {frame + 1}/{n_frames}")

    plt.close(fig)
    print(f"  ✓ atom_orbit.mp4 ({n_frames} frames)")


def render_radial_comparison(
    radial_r: np.ndarray,
    radial_rho: np.ndarray,
    radial_P: np.ndarray,
    analytic_rho: np.ndarray,
    output_dir: str,
) -> None:
    if not VIZ_AVAILABLE:
        print("⚠️  Radial comparison skipped (matplotlib not available)")
        return

    print("\nRendering radial comparison plot...")
    out_path = Path(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=300)

    axes[0].plot(radial_r, radial_rho, "b-", linewidth=2, label="Numeric")
    axes[0].plot(radial_r, analytic_rho, "r--", linewidth=2, label="Analytic (1s)")
    axes[0].set_xlabel("r (a.u.)")
    axes[0].set_ylabel("ρ(r)")
    axes[0].set_title("Radial density")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, min(10.0, radial_r.max()))

    analytic_P = 4.0 * math.pi * (radial_r ** 2) * analytic_rho
    axes[1].plot(radial_r, radial_P, "b-", linewidth=2, label="Numeric")
    axes[1].plot(radial_r, analytic_P, "r--", linewidth=2, label="Analytic (1s)")
    axes[1].set_xlabel("r (a.u.)")
    axes[1].set_ylabel("P(r) = 4πr²ρ(r)")
    axes[1].set_title("Radial probability distribution")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, min(10.0, radial_r.max()))

    plt.tight_layout()
    fig.savefig(out_path / "radial_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ radial_compare.png")


def generate_all_visualizations(results: Dict[str, object], output_dir: str) -> None:
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    density = results["density"]
    L = float(results["config"]["L"])

    render_orthogonal_slices(density, L, output_dir, dpi=1000)
    render_isosurfaces(density, L, output_dir, levels=(0.6, 0.3, 0.1), dpi=1000)
    render_orbit_video(density, L, output_dir, n_frames=720, fps=30, dpi=480)
    render_radial_comparison(
        results["radial_r"],
        results["radial_rho"],
        results["radial_P"],
        results["analytic_rho"],
        output_dir,
    )

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main CLI entry
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="REAL ATOM 3D DISCOVERY — FULL 4K VERSION (V2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=DEFAULT_CONFIG["N"], help="Grid size (N×N×N)")
    parser.add_argument("--L", type=float, default=DEFAULT_CONFIG["L"], help="Box size (atomic units)")
    parser.add_argument("--Z", type=float, default=DEFAULT_CONFIG["Z"], help="Nuclear charge")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["steps"], help="Evolution steps")
    parser.add_argument("--dt", type=float, default=DEFAULT_CONFIG["dt"], help="Imaginary-time step size")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_CONFIG["ε"],
        help="Nuclear softening parameter ε",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_CONFIG["save_every"],
        help="Record energy / metrics every N steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/real_atom_3d_v2",
        help="Where to store simulation artifacts",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization rendering")

    args = parser.parse_args()

    config = {
        "Z": args.Z,
        "L": args.L,
        "ε": args.epsilon,
        "N": args.N,
        "steps": args.steps,
        "dt": args.dt,
        "save_every": args.save_every,
        "seed": args.seed,
        "facts_only": True,
    }

    print("\n" + "=" * 70)
    print("REAL ATOM 3D DISCOVERY — FULL 4K VERSION (V2)")
    print("=" * 70)
    print("Physics-only ground-state discovery via imaginary-time propagation")
    print("No analytic orbitals. No symmetry enforcement. Pure PDE results.")
    print("=" * 70)

    print_hardware_info()
    auto_adjust_grid_for_memory(config)

    try:
        results = imaginary_time_propagation(config)
    except MemoryError:
        if config["N"] > 256:
            print("\n❌ MemoryError encountered. Retrying with N=256...")
            config["N"] = 256
            results = imaginary_time_propagation(config)
        else:
            print("\n❌ MemoryError even at N=256. Please reduce N further or add RAM.")
            return

    save_artifacts(results, args.output_dir)

    if not args.no_viz:
        generate_all_visualizations(results, args.output_dir)

    print("\n" + "=" * 70)
    print("✅ REAL-ATOM-3D-V2 COMPLETE")
    print("=" * 70)
    print(f"\nArtifacts written to {args.output_dir}/")
    print("\nKey images:")
    print("  • atom_xy_4k.png")
    print("  • atom_xz_4k.png")
    print("  • atom_yz_4k.png")
    if ISOSURFACE_AVAILABLE:
        print("  • isosurface_0.3.png")
    if VIDEO_AVAILABLE and ISOSURFACE_AVAILABLE:
        print("  • atom_orbit.mp4")
    print("  • radial_compare.png")
    print(
        f"\nFinal energy ≈ {results['E_final']:.4f} H"
        " (hydrogenic reference ≈ -0.5 H)"
    )
    print(f"χ² deviation from analytic 1s density: {results['chi_squared']:.6e}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
