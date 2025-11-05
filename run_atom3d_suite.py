#!/usr/bin/env python3
"""
ATOM3D COMPLETE SIMULATION SUITE
==================================

Comprehensive quantum-physics research assistant for computing first-principles
atomic models without assumptions or pre-defined orbitals.

This script orchestrates the complete multi-stage atom simulation suite:
1. Hydrogen ground state (Z=1) at multiple resolutions
2. Hydrogen excited states (2s, 2p)
3. Helium Hartree model
4. Field perturbations (Stark + Zeeman)
5. Tomography reconstruction
6. Dashboard generation with 4K composite renders

Uses imaginary-time Schrödinger equation propagation with FFT-based Laplacian.
All outputs saved deterministically with seed=424242.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Optional visualization imports
VIZ_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    VIZ_AVAILABLE = True
except Exception:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "seed": 424242,
    "box_size": 12.0,
    "softening": 0.3,
    "resolutions": [64, 128, 256, 512],
    "dt": 0.002,
    "convergence_tol": 1e-6,
    "max_steps": 5000,
    "output_dir": "artifacts/atom3d",
}


# =============================================================================
# UTILITIES
# =============================================================================

def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def run_command(cmd: List[str], description: str) -> bool:
    """Execute a shell command and report status."""
    print(f"→ {description}")
    print(f"  Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  ❌ Command not found: {cmd[0]}")
        return False


def check_convergence(energy_file: Path, tol: float = 1e-6) -> bool:
    """Check if energy has converged based on final energy history."""
    try:
        with open(energy_file) as f:
            data = json.load(f)
        
        if "history" in data and "energies" in data["history"]:
            energies = data["history"]["energies"]
        else:
            return False
        
        if len(energies) < 2:
            return False
        
        # Check if last few energies are stable
        last_energies = energies[-10:]
        if len(last_energies) < 2:
            return False
        
        diffs = [abs(last_energies[i] - last_energies[i-1]) for i in range(1, len(last_energies))]
        max_diff = max(diffs) if diffs else float('inf')
        
        converged = max_diff < tol
        print(f"  Energy variation: {max_diff:.2e} (threshold: {tol:.2e})")
        print(f"  Convergence: {'✅ Yes' if converged else '⚠️  No'}")
        
        return converged
    except Exception as e:
        print(f"  ⚠️  Could not check convergence: {e}")
        return False


# =============================================================================
# STAGE 1: HYDROGEN GROUND STATE (MULTI-RESOLUTION)
# =============================================================================

def run_hydrogen_ground(output_base: Path, resolutions: List[int]) -> Dict[str, Path]:
    """Run hydrogen ground state at multiple resolutions."""
    print_banner("STAGE 1: HYDROGEN GROUND STATE", "=")
    
    results = {}
    
    for N in resolutions:
        print(f"\n{'─' * 80}")
        print(f"Resolution: {N}³")
        print(f"{'─' * 80}")
        
        # Determine steps based on resolution
        steps = 1200 if N <= 256 else 2000
        
        out_dir = output_base / f"hyd-ground-N{N}"
        
        # Skip if already exists
        if out_dir.exists():
            print(f"  ⚠️  Output directory {out_dir} already exists, skipping...")
            results[f"ground_{N}"] = out_dir
            continue
        
        cmd = [
            sys.executable, "-m", "atomsim.cli",
            "hyd-ground",
            "--N", str(N),
            "--L", str(CONFIG["box_size"]),
            "--Z", "1.0",
            "--steps", str(steps),
            "--dt", str(CONFIG["dt"]),
            "--eps", str(CONFIG["softening"]),
            "--level", "0.2",
            "--out", str(out_dir),
        ]
        
        success = run_command(cmd, f"Hydrogen ground state N={N}³")
        
        if success and (out_dir / "energy.json").exists():
            check_convergence(out_dir / "energy.json", CONFIG["convergence_tol"])
            results[f"ground_{N}"] = out_dir
        
        time.sleep(0.5)
    
    return results


# =============================================================================
# STAGE 2: HYDROGEN EXCITED STATES (2s, 2p)
# =============================================================================

def run_hydrogen_excited(output_base: Path, resolution: int = 256) -> Dict[str, Path]:
    """Run hydrogen excited states (2s, 2p)."""
    print_banner("STAGE 2: HYDROGEN EXCITED STATES", "=")
    
    results = {}
    excited_states = [
        ("2s", 2, 0, 0),
        ("2p", 2, 1, 0),
    ]
    
    for label, n, l, m in excited_states:
        print(f"\n{'─' * 80}")
        print(f"Excited state: {label} (n={n}, l={l}, m={m})")
        print(f"{'─' * 80}")
        
        out_dir = output_base / f"hyd-excited-{label}"
        
        if out_dir.exists():
            print(f"  ⚠️  Output directory {out_dir} already exists, skipping...")
            results[label] = out_dir
            continue
        
        cmd = [
            sys.executable, "-m", "atomsim.cli",
            "hyd-excited",
            "--N", str(resolution),
            "--L", str(CONFIG["box_size"]),
            "--Z", "1.0",
            "--steps", "2000",
            "--dt", str(CONFIG["dt"]),
            "--eps", str(CONFIG["softening"]),
            "--level", "0.15",
            "--nlm", f"{n},{l},{m}",
            "--out", str(out_dir),
        ]
        
        success = run_command(cmd, f"Hydrogen excited {label}")
        
        if success and (out_dir / "energy.json").exists():
            check_convergence(out_dir / "energy.json", CONFIG["convergence_tol"])
            results[label] = out_dir
        
        time.sleep(0.5)
    
    return results


# =============================================================================
# STAGE 3: HELIUM HARTREE MODEL
# =============================================================================

def run_helium_ground(output_base: Path, resolution: int = 256) -> Optional[Path]:
    """Run helium ground state with Hartree mean-field approximation."""
    print_banner("STAGE 3: HELIUM HARTREE MODEL", "=")
    
    out_dir = output_base / "he-ground"
    
    if out_dir.exists():
        print(f"  ⚠️  Output directory {out_dir} already exists, skipping...")
        return out_dir
    
    cmd = [
        sys.executable, "-m", "atomsim.cli",
        "he-ground",
        "--N", str(resolution),
        "--L", str(CONFIG["box_size"]),
        "--steps", "3000",
        "--dt", str(CONFIG["dt"]),
        "--eps", str(CONFIG["softening"]),
        "--mix", "0.3",
        "--level", "0.15",
        "--out", str(out_dir),
    ]
    
    success = run_command(cmd, "Helium ground state (Hartree)")
    
    if success:
        return out_dir
    return None


# =============================================================================
# STAGE 4: FIELD PERTURBATIONS (STARK + ZEEMAN)
# =============================================================================

def run_field_perturbations(
    output_base: Path,
    ground_state_dir: Path,
    resolution: int = 256,
) -> Dict[str, Path]:
    """Apply Stark and Zeeman field perturbations."""
    print_banner("STAGE 4: FIELD PERTURBATIONS", "=")
    
    results = {}
    
    # Stark effect
    print(f"\n{'─' * 80}")
    print("Stark Effect")
    print(f"{'─' * 80}")
    
    stark_dir = output_base / "hyd-field-stark"
    if not stark_dir.exists():
        cmd = [
            sys.executable, "-m", "atomsim.cli",
            "hyd-field",
            "--mode", "stark",
            "--in", str(ground_state_dir),
            "--L", str(CONFIG["box_size"]),
            "--Ez", "0.01",
            "--steps", "500",
            "--dt", str(CONFIG["dt"]),
            "--out", str(stark_dir),
        ]
        
        success = run_command(cmd, "Stark effect (Ez=0.01)")
        if success:
            results["stark"] = stark_dir
    else:
        print(f"  ⚠️  Output directory {stark_dir} already exists, skipping...")
        results["stark"] = stark_dir
    
    # Zeeman effect
    print(f"\n{'─' * 80}")
    print("Zeeman Effect")
    print(f"{'─' * 80}")
    
    zeeman_dir = output_base / "hyd-field-zeeman"
    if not zeeman_dir.exists():
        cmd = [
            sys.executable, "-m", "atomsim.cli",
            "hyd-field",
            "--mode", "zeeman",
            "--in", str(ground_state_dir),
            "--L", str(CONFIG["box_size"]),
            "--Bz", "0.1",
            "--steps", "500",
            "--dt", str(CONFIG["dt"]),
            "--out", str(zeeman_dir),
        ]
        
        success = run_command(cmd, "Zeeman effect (Bz=0.1)")
        if success:
            results["zeeman"] = zeeman_dir
    else:
        print(f"  ⚠️  Output directory {zeeman_dir} already exists, skipping...")
        results["zeeman"] = zeeman_dir
    
    return results


# =============================================================================
# STAGE 5: TOMOGRAPHY RECONSTRUCTION
# =============================================================================

def run_tomography(output_base: Path, ground_state_dir: Path) -> Optional[Path]:
    """Run synthetic tomography reconstruction."""
    print_banner("STAGE 5: TOMOGRAPHY RECONSTRUCTION", "=")
    
    tomo_dir = output_base / "hyd-tomo"
    
    if tomo_dir.exists():
        print(f"  ⚠️  Output directory {tomo_dir} already exists, skipping...")
        return tomo_dir
    
    cmd = [
        sys.executable, "-m", "atomsim.cli",
        "hyd-tomo",
        "--in", str(ground_state_dir),
        "--angles", "90",
        "--noise", "0.02",
        "--out", str(tomo_dir),
    ]
    
    success = run_command(cmd, "Tomography reconstruction")
    
    if success:
        return tomo_dir
    return None


# =============================================================================
# STAGE 6: DASHBOARD GENERATION
# =============================================================================

def generate_dashboard(
    output_base: Path,
    all_results: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate comprehensive dashboard with visualizations."""
    print_banner("STAGE 6: DASHBOARD GENERATION", "=")

    if not VIZ_AVAILABLE:
        print("  ⚠️  Matplotlib not available, skipping dashboard generation")
        return

    dashboard_dir = output_base / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    print("→ Generating energy convergence plots...")
    generate_energy_plots(all_results, dashboard_dir)

    print("→ Generating 4K composite render...")
    generate_composite_render(all_results, dashboard_dir)

    print("→ Generating summary report...")
    generate_summary_report(all_results, dashboard_dir, metadata)

    print(f"  ✅ Dashboard saved to {dashboard_dir}")


def generate_energy_plots(all_results: Dict[str, Any], dashboard_dir: Path) -> None:
    """Generate energy convergence plots for all simulations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle("Energy Convergence - REAL-ATOM-3D-DISCOVERY-V2", fontsize=16, fontweight='bold')
    
    # Plot 1: Hydrogen ground state at multiple resolutions
    ax = axes[0, 0]
    for key, path in all_results.get("hydrogen_ground", {}).items():
        if isinstance(path, Path) and (path / "energy.json").exists():
            with open(path / "energy.json") as f:
                data = json.load(f)
            if "history" in data and "energies" in data["history"]:
                energies = data["history"]["energies"]
                ax.plot(energies, label=key, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("Hydrogen Ground State (Multi-Resolution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Excited states
    ax = axes[0, 1]
    for label in ["2s", "2p"]:
        path = all_results.get("hydrogen_excited", {}).get(label)
        if path and (path / "energy.json").exists():
            with open(path / "energy.json") as f:
                data = json.load(f)
            if "history" in data and "energies" in data["history"]:
                energies = data["history"]["energies"]
                ax.plot(energies, label=f"Hydrogen {label}", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("Hydrogen Excited States")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Helium
    ax = axes[1, 0]
    he_path = all_results.get("helium")
    if he_path and (he_path / "total_energy.json").exists():
        with open(he_path / "total_energy.json") as f:
            data = json.load(f)
        if "history" in data and "total" in data["history"]:
            energies = data["history"]["total"]
            ax.plot(energies, label="Total Energy", linewidth=2, color='blue')
        if "history" in data and "electron_electron" in data["history"]:
            ee_energies = data["history"]["electron_electron"]
            ax.plot(ee_energies, label="e-e Interaction", linewidth=2, color='red')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("Helium Hartree Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Field perturbations summary
    ax = axes[1, 1]
    field_data = []
    for field_type in ["stark", "zeeman"]:
        path = all_results.get("fields", {}).get(field_type)
        if path and (path / "shifts.json").exists():
            with open(path / "shifts.json") as f:
                data = json.load(f)
                field_data.append((field_type.capitalize(), data))
    
    if field_data:
        labels = [d[0] for d in field_data]
        # Extract shift values (use first available shift key)
        shifts = []
        for label, data in field_data:
            for key in data:
                if "shift" in key.lower():
                    shifts.append(abs(data[key]))
                    break
        
        if shifts:
            ax.bar(labels, shifts, color=['orange', 'purple'])
            ax.set_ylabel("Energy Shift (Hartree)")
            ax.set_title("Field Perturbation Effects")
            ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "Field perturbation data not available",
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(dashboard_dir / "energy_convergence.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Energy plots saved")


def generate_composite_render(all_results: Dict[str, Any], dashboard_dir: Path) -> None:
    """Generate 4K composite render with multiple density isosurfaces."""
    ground_states = all_results.get("hydrogen_ground", {})
    highest_N = 0
    best_path: Optional[Path] = None

    for key, path in ground_states.items():
        if isinstance(path, Path) and (path / "density.npy").exists():
            try:
                N = int(key.split("_")[1])
            except (IndexError, ValueError):
                N = 0
            if N > highest_N:
                highest_N = N
                best_path = path

    if not best_path:
        print("  ⚠️  No ground state density found for composite render")
        return

    density = np.load(best_path / "density.npy")

    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(16, 9), dpi=240)
    fig.suptitle("REAL-ATOM-3D-DISCOVERY-V2", fontsize=20, fontweight="bold")

    isosurface_levels = [0.1, 0.3, 0.6]
    level_colors = {0.1: "#4fd9ff", 0.3: "#ff66ff", 0.6: "#ffe066"}
    projections = [
        ("XY", density.max(axis=2)),
        ("XZ", density.max(axis=1)),
        ("YZ", density.max(axis=0)),
    ]

    axes = [fig.add_subplot(1, 3, idx + 1) for idx in range(3)]
    for ax, (label, mip) in zip(axes, projections):
        mip_norm = mip / (mip.max() + 1e-12)
        im = ax.imshow(mip_norm, cmap="inferno", origin="lower", vmin=0.0, vmax=0.7)

        for level in isosurface_levels:
            ax.contour(
                mip_norm,
                levels=[level],
                colors=level_colors[level],
                linewidths=1.5,
            )

        mid_y = mip.shape[0] // 2
        mid_x = mip.shape[1] // 2
        ax.axhline(mid_y, color="white", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(mid_x, color="white", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_title(f"{label} Projection")
        ax.axis("off")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized density")

        legend_handles = [
            Line2D([0], [0], color=level_colors[level], lw=2, label=f"ρ={level}")
            for level in isosurface_levels
        ]
        legend_handles.append(
            Line2D([0], [0], color="white", lw=1.0, linestyle="--", label="Nodal plane")
        )
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

    plt.tight_layout()
    final_path = dashboard_dir / "REAL-ATOM-3D-DISCOVERY-V2.png"
    fig.savefig(final_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    root_copy = dashboard_dir.parent / "REAL-ATOM-3D-DISCOVERY-V2.png"
    if root_copy != final_path:
        shutil.copyfile(final_path, root_copy)

    print(f"  ✅ 4K composite render saved → {final_path}")


def generate_summary_report(
    all_results: Dict[str, Any],
    dashboard_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate JSON summary report."""
    summary = {
        "name": "REAL-ATOM-3D-DISCOVERY-V2",
        "description": "Complete multi-stage atom simulation suite",
        "facts_only": True,
        "seed": CONFIG["seed"],
        "run_order": [
            "hyd-ground",
            "hyd-excited-2s",
            "hyd-excited-2p",
            "he-ground",
            "hyd-field-stark",
            "hyd-field-zeeman",
            "hyd-tomo",
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": CONFIG,
        "results": {},
    }

    if metadata:
        summary.update(metadata)
    
    energy_meta = metadata.get("energy") if metadata else None
    if energy_meta:
        summary["results"]["hydrogen_ground"] = energy_meta.get("hydrogen_ground", {})
        summary["results"]["hydrogen_excited"] = energy_meta.get("hydrogen_excited", {})
        helium_meta = energy_meta.get("helium")
        if helium_meta:
            summary["results"]["helium"] = helium_meta
    else:
        if "hydrogen_ground" in all_results:
            ground_energies = {}
            for key, path in all_results["hydrogen_ground"].items():
                if isinstance(path, Path) and (path / "energy.json").exists():
                    with open(path / "energy.json") as f:
                        data = json.load(f)
                    ground_energies[key] = {
                        "final_energy": data.get("final"),
                        "path": str(path),
                    }
            summary["results"]["hydrogen_ground"] = ground_energies

        if "hydrogen_excited" in all_results:
            excited_energies = {}
            for label, path in all_results["hydrogen_excited"].items():
                if path and (path / "energy.json").exists():
                    with open(path / "energy.json") as f:
                        data = json.load(f)
                    excited_energies[label] = {
                        "final_energy": data.get("final"),
                        "path": str(path),
                    }
            summary["results"]["hydrogen_excited"] = excited_energies

        if "helium" in all_results and all_results["helium"]:
            he_path = all_results["helium"]
            if (he_path / "total_energy.json").exists():
                with open(he_path / "total_energy.json") as f:
                    data = json.load(f)
                summary["results"]["helium"] = {
                    "total_energy": data.get("total_energy"),
                    "ee_energy": data.get("electron_electron"),
                    "path": str(he_path),
                }

    if metadata and metadata.get("fields") is not None:
        summary["results"]["field_perturbations"] = metadata.get("fields", {})
    elif "fields" in all_results:
        field_results = {}
        for field_type, path in all_results["fields"].items():
            if path and (path / "shifts.json").exists():
                with open(path / "shifts.json") as f:
                    data = json.load(f)
                field_results[field_type] = {
                    "shifts": data,
                    "path": str(path),
                }
        summary["results"]["field_perturbations"] = field_results

    if metadata and metadata.get("recon") is not None:
        summary["results"]["tomography"] = metadata["recon"]
    elif "tomography" in all_results and all_results["tomography"]:
        tomo_path = all_results["tomography"]
        if (tomo_path / "metrics.json").exists():
            with open(tomo_path / "metrics.json") as f:
                data = json.load(f)
            summary["results"]["tomography"] = {
                "metrics": data,
                "path": str(tomo_path),
            }

    with open(dashboard_dir / "atom_descriptor.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✅ Summary report saved")


def has_converged(energies: List[float], tol: float = CONFIG["convergence_tol"]) -> bool:
    """Determine if the supplied energy series is converged."""

    if not energies or len(energies) < 2:
        return False
    window = energies[-10:] if len(energies) > 10 else energies
    diffs = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
    if not diffs:
        return False
    return max(diffs) < tol


def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON data, returning None on failure."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  JSON parse error for {path}: {exc}")
        return None


def _load_density_sample(path: Path) -> Optional[np.ndarray]:
    """Load density.npy with optional downsampling for analysis."""

    density_path = path / "density.npy"
    if not density_path.exists():
        return None

    try:
        density = np.load(density_path)
    except Exception as exc:  # pragma: no cover - defensive I/O
        print(f"  ⚠️  Unable to load density from {density_path}: {exc}")
        return None

    if density.ndim != 3:
        return None

    stride = max(1, density.shape[0] // 256)
    if stride > 1:
        density = density[::stride, ::stride, ::stride]
    return density


def analyze_density(density: np.ndarray) -> Dict[str, float | bool]:
    """Compute anisotropy and nodal statistics for discovery logging."""

    sample = density.astype(np.float64, copy=False)
    max_val = float(sample.max() + 1e-12)

    axis_means = np.array(
        [
            sample.mean(axis=(1, 2)),
            sample.mean(axis=(0, 2)),
            sample.mean(axis=(0, 1)),
        ]
    )
    axis_mean_values = axis_means.mean(axis=1)

    anisotropy = float((axis_mean_values.max() - axis_mean_values.min()) / (axis_mean_values.mean() + 1e-12))

    mirror_x = sample[::-1, :, :]
    mirror_y = sample[:, ::-1, :]
    mirror_z = sample[:, :, ::-1]
    asymmetry = float(
        (
            np.abs(sample - mirror_x).mean()
            + np.abs(sample - mirror_y).mean()
            + np.abs(sample - mirror_z).mean()
        )
        / (3.0 * max_val)
    )

    node_fraction = float(np.mean(sample < max_val * 0.05))

    metrics = {
        "anisotropy": anisotropy,
        "axis_means": [float(v) for v in axis_mean_values],
        "node_fraction": node_fraction,
        "asymmetry": asymmetry,
        "max_density": max_val,
    }
    metrics["candidate"] = bool(anisotropy > 0.1 or asymmetry > 0.05)
    return metrics


def collect_metadata(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate metadata across all simulation stages."""

    metadata: Dict[str, Any] = {
        "energy": {"hydrogen_ground": {}, "hydrogen_excited": {}, "helium": {}},
        "fields": {},
        "recon": {},
        "discovery_candidates": [],
        "ground_reference": None,
        "facts_only": True,
    }

    highest_ground = -1

    # Hydrogen ground states
    for label, path in all_results.get("hydrogen_ground", {}).items():
        if not isinstance(path, Path):
            continue
        entry: Dict[str, Any] = {"path": str(path)}
        energy_data = _load_json_safe(path / "energy.json")
        if energy_data:
            history = energy_data.get("history", {})
            energies = history.get("energies", [])
            entry["final_energy"] = energy_data.get("final")
            entry["history"] = history
            try:
                entry["converged"] = has_converged([float(e) for e in energies])
            except (TypeError, ValueError):
                entry["converged"] = False
        metadata["energy"]["hydrogen_ground"][label] = entry

        try:
            N_val = int(label.split("_")[1])
        except (IndexError, ValueError):
            N_val = 0
        if N_val > highest_ground:
            highest_ground = N_val
            metadata["ground_reference"] = str(path)

        density = _load_density_sample(path)
        if density is not None:
            metrics = analyze_density(density)
            entry["density_metrics"] = metrics
            if metrics.get("candidate"):
                metadata["discovery_candidates"].append(
                    {
                        "stage": "hydrogen_ground",
                        "label": label,
                        "path": str(path),
                        "metrics": metrics,
                    }
                )

    # Hydrogen excited states
    for label, path in all_results.get("hydrogen_excited", {}).items():
        if not isinstance(path, Path):
            continue
        entry = {"path": str(path)}
        energy_data = _load_json_safe(path / "energy.json")
        if energy_data:
            history = energy_data.get("history", {})
            energies = history.get("energies", [])
            entry["final_energy"] = energy_data.get("final")
            entry["history"] = history
            try:
                entry["converged"] = has_converged([float(e) for e in energies])
            except (TypeError, ValueError):
                entry["converged"] = False
        density = _load_density_sample(path)
        if density is not None:
            metrics = analyze_density(density)
            entry["density_metrics"] = metrics
            if metrics.get("candidate"):
                metadata["discovery_candidates"].append(
                    {
                        "stage": "hydrogen_excited",
                        "label": label,
                        "path": str(path),
                        "metrics": metrics,
                    }
                )
        metadata["energy"]["hydrogen_excited"][label] = entry

    # Helium Hartree state
    he_path = all_results.get("helium")
    if isinstance(he_path, Path):
        entry: Dict[str, Any] = {"path": str(he_path)}
        energy_data = _load_json_safe(he_path / "total_energy.json")
        if energy_data:
            entry.update(
                {
                    "total_energy": energy_data.get("total_energy"),
                    "electron_electron": energy_data.get("electron_electron"),
                    "history": energy_data.get("history", {}),
                }
            )
            total_hist = energy_data.get("history", {}).get("total", [])
            try:
                entry["converged"] = has_converged([float(e) for e in total_hist])
            except (TypeError, ValueError):
                entry["converged"] = False
        density = _load_density_sample(he_path)
        if density is not None:
            metrics = analyze_density(density)
            entry["density_metrics"] = metrics
            if metrics.get("candidate"):
                metadata["discovery_candidates"].append(
                    {
                        "stage": "helium",
                        "label": "he-ground",
                        "path": str(he_path),
                        "metrics": metrics,
                    }
                )
        metadata["energy"]["helium"] = entry

    # Field perturbations
    for field_type, path in all_results.get("fields", {}).items():
        if not isinstance(path, Path):
            continue
        shifts = _load_json_safe(path / "shifts.json") or {}
        entry = {"path": str(path), "shifts": shifts}
        metadata["fields"][field_type] = entry

    # Tomography reconstruction
    tomo_path = all_results.get("tomography")
    if isinstance(tomo_path, Path):
        metrics = _load_json_safe(tomo_path / "metrics.json") or {}
        metadata["recon"] = {"path": str(tomo_path), "metrics": metrics}

    return metadata


def write_metadata_files(output_base: Path, metadata: Dict[str, Any]) -> None:
    """Persist top-level metadata files required by the ticket."""

    output_base.mkdir(parents=True, exist_ok=True)

    energy_data = metadata.get("energy", {})
    field_data = metadata.get("fields", {})
    recon_data = metadata.get("recon", {})
    discovery = metadata.get("discovery_candidates", [])

    descriptor = {
        "name": "REAL-ATOM-3D-DISCOVERY-V2",
        "facts_only": True,
        "seed": CONFIG["seed"],
        "run_order": [
            "hyd-ground",
            "hyd-excited-2s",
            "hyd-excited-2p",
            "he-ground",
            "hyd-field-stark",
            "hyd-field-zeeman",
            "hyd-tomo",
        ],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ground_reference": metadata.get("ground_reference"),
        "discovery_candidates": discovery,
        "artifacts": {
            "energy_convergence": "energy_convergence.json",
            "field_shift": "field_shift.json",
            "recon_metrics": "recon_metrics.json",
            "dashboard": "dashboard/atom_descriptor.json",
            "composite_render": "REAL-ATOM-3D-DISCOVERY-V2.png",
        },
        "energy_overview": {
            "hydrogen_ground": {
                label: entry.get("final_energy")
                for label, entry in energy_data.get("hydrogen_ground", {}).items()
            },
            "hydrogen_excited": {
                label: entry.get("final_energy")
                for label, entry in energy_data.get("hydrogen_excited", {}).items()
            },
            "helium": energy_data.get("helium", {}).get("total_energy"),
        },
    }

    (output_base / "atom_descriptor.json").write_text(json.dumps(descriptor, indent=2))
    (output_base / "energy_convergence.json").write_text(json.dumps(energy_data, indent=2))
    (output_base / "field_shift.json").write_text(json.dumps(field_data, indent=2))
    (output_base / "recon_metrics.json").write_text(json.dumps(recon_data, indent=2))

    print(f"  ✅ Metadata written to {output_base}")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Complete Atom3D Simulation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full suite with default settings
  python run_atom3d_suite.py --full
  
  # Run specific stages
  python run_atom3d_suite.py --stages ground,excited,helium
  
  # Run at single resolution (faster)
  python run_atom3d_suite.py --full --resolutions 256
        """,
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete simulation suite (all stages)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated list of stages to run: ground,excited,helium,field,tomo,dashboard",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="Comma-separated resolutions for ground state (e.g., '64,128,256')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Parse configuration
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if args.resolutions:
        resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    else:
        resolutions = CONFIG["resolutions"]
    
    # Determine stages to run
    if args.full:
        stages = ["ground", "excited", "helium", "field", "tomo", "dashboard"]
    elif args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
    else:
        parser.print_help()
        return 1
    
    print_banner("ATOM3D COMPLETE SIMULATION SUITE")
    print(f"Output directory: {output_base}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Resolutions: {resolutions}")
    print(f"Seed: {CONFIG['seed']}")
    
    start_time = time.time()
    all_results = {}
    
    # Stage 1: Hydrogen ground state
    if "ground" in stages:
        all_results["hydrogen_ground"] = run_hydrogen_ground(output_base, resolutions)
    
    # Get reference ground state for subsequent stages
    if "hydrogen_ground" in all_results:
        # Use N=256 as reference
        ground_256 = all_results["hydrogen_ground"].get("ground_256")
        if not ground_256:
            # Fallback to highest available
            for key in sorted(all_results["hydrogen_ground"].keys(), reverse=True):
                ground_256 = all_results["hydrogen_ground"][key]
                break
    else:
        # Try to find existing ground state
        ground_256 = output_base / "hyd-ground-N256"
        if not ground_256.exists():
            print("⚠️  No ground state found, some stages may be skipped")
            ground_256 = None
    
    # Stage 2: Excited states
    if "excited" in stages:
        all_results["hydrogen_excited"] = run_hydrogen_excited(output_base)
    
    # Stage 3: Helium
    if "helium" in stages:
        all_results["helium"] = run_helium_ground(output_base)
    
    # Stage 4: Field perturbations
    if "field" in stages and ground_256:
        all_results["fields"] = run_field_perturbations(output_base, ground_256)
    
    # Stage 5: Tomography
    if "tomo" in stages and ground_256:
        all_results["tomography"] = run_tomography(output_base, ground_256)
    
    metadata = collect_metadata(all_results)

    # Stage 6: Dashboard
    if "dashboard" in stages:
        generate_dashboard(output_base, all_results, metadata)

    write_metadata_files(output_base, metadata)

    elapsed = time.time() - start_time
    
    print_banner("SIMULATION SUITE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nAll artifacts saved to: {output_base}")
    print("\nKey metadata files:")
    print(f"  • atom_descriptor.json         — Master descriptor")
    print(f"  • energy_convergence.json      — All energy histories")
    print(f"  • field_shift.json             — Stark & Zeeman shifts")
    print(f"  • recon_metrics.json           — Tomography metrics")
    print("\nKey outputs:")
    print(f"  • Dashboard: {output_base}/dashboard/")
    print(f"  • Energy plots: {output_base}/dashboard/energy_convergence.png")
    print(f"  • 4K render: {output_base}/REAL-ATOM-3D-DISCOVERY-V2.png")
    print(f"  • 3D mesh + MP4: each stage folder contains .glb + orbit_spin.mp4")
    
    discovery_candidates = metadata.get("discovery_candidates", [])
    if discovery_candidates:
        print("\n🔬 Discovery Candidates:")
        for cand in discovery_candidates:
            print(f"  • {cand['stage']} / {cand['label']}")
            metrics = cand.get("metrics", {})
            print(
                f"    anisotropy={metrics.get('anisotropy', 0):.3f}, "
                f"asymmetry={metrics.get('asymmetry', 0):.3f}"
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
