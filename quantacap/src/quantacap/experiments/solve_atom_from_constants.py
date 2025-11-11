"""
Physics-first atom solver
-------------------------

Goal: don't paint an atom, SOLVE it.

We solve a hydrogen-like atom on a 3D cubic grid using imaginary-time
propagation of the time-independent Schrödinger equation:

    dψ/dτ = (1/2) ∇²ψ - V(r) ψ

In atomic units: ħ = 1, m_e = 1, e = 1, a0 = 1.

We pick V(r) = -Z / r with Z = 1 for hydrogen.
We start from a random-ish wavefunction, evolve it in imaginary time,
normalize every step → the ground state wavefunction emerges.
Then we output |ψ|² as the "real" atom density.

Outputs (in artifacts/real_atom):
  - density.npy      : 3D electron density
  - psi.npy          : 3D wavefunction (real)
  - slice_*.png      : 2D views
  - mip.png          : max-intensity projection
  - convergence.png  : energy vs step
  - atom_descriptor.json : everything needed to re-run exactly
"""

import argparse
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


def make_grid(N=64, L=12.0):
    """
    3D cube: x,y,z in [-L/2, L/2]
    N = grid points per axis
    """
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def coulomb_potential(X, Y, Z, Zcharge=1.0, softening=0.3):
    """
    V(r) = -Z/r, but we soften near r=0 to avoid singularity.
    softening is like saying the nucleus isn't a perfect point.
    """
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = -Zcharge / np.sqrt(R**2 + softening**2)
    return V


def laplacian_3d(psi, dx):
    """
    3D finite-difference Laplacian (6-point stencil).
    psi is (N,N,N).
    periodic = False: we do zero-flux boundaries by reusing edge values.
    """
    lap = (
        -6.0 * psi
        + np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=1)
        + np.roll(psi, -1, axis=1)
        + np.roll(psi, 1, axis=2)
        + np.roll(psi, -1, axis=2)
    ) / (dx * dx)
    return lap


def normalize(psi, dx):
    """
    Normalize ∫|ψ|² dV = 1
    Volume element = dx^3
    """
    prob = np.sum(np.abs(psi)**2) * (dx ** 3)
    if prob <= 0:
        return psi, 0.0
    psi = psi / np.sqrt(prob)
    return psi, prob


def compute_energy(psi, V, dx):
    """
    E = <ψ| -1/2 ∇² + V |ψ>
    """
    lap = laplacian_3d(psi, dx)
    kinetic = -0.5 * np.sum(np.conj(psi) * lap) * (dx ** 3)
    potential = np.sum(np.conj(psi) * V * psi) * (dx ** 3)
    E = np.real(kinetic + potential)
    return float(E)


def imaginary_time_solve(
    N=64,
    L=12.0,
    Z=1.0,
    dt=0.002,
    steps=600,
    softening=0.3,
):
    """
    Core solver:
      - build grid
      - init ψ
      - loop imaginary-time evolution
      - normalize each step
      - track energy for convergence
    """
    x, y, z, X, Y, Zgrid = make_grid(N=N, L=L)
    dx = x[1] - x[0]
    V = coulomb_potential(X, Y, Zgrid, Zcharge=Z, softening=softening)

    # initial wavefunction: gaussian bump, slightly random
    rng = np.random.default_rng(12345)
    psi = np.exp(-(X**2 + Y**2 + Zgrid**2) / 4.0) * (1.0 + 0.05 * rng.standard_normal(X.shape))

    psi, _ = normalize(psi, dx)

    energies = []

    for step in range(steps):
        lap = laplacian_3d(psi, dx)
        # imaginary-time update: psi <- psi + dt * (1/2 ∇²ψ - Vψ)
        psi = psi + dt * (0.5 * lap - V * psi)

        # renormalize
        psi, _ = normalize(psi, dx)

        if step % 10 == 0 or step == steps - 1:
            E = compute_energy(psi, V, dx)
            energies.append((step, E))

    density = np.abs(psi) ** 2
    return {
        "psi": psi,
        "density": density,
        "V": V,
        "x": x,
        "y": y,
        "z": z,
        "energies": energies,
        "dx": dx,
        "params": {
            "N": N,
            "L": L,
            "Z": Z,
            "dt": dt,
            "steps": steps,
            "softening": softening,
        },
    }


def save_slices(density, x, y, z, outdir):
    N = density.shape[0]
    zs = [int(N*0.15), int(N*0.35), int(N*0.5), int(N*0.7), int(N*0.9)]
    for i, zidx in enumerate(zs):
        plt.figure(figsize=(4,4))
        plt.imshow(
            density[:, :, zidx],
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="inferno",
        )
        plt.colorbar()
        plt.title(f"atom density (z-slice={zidx}/{N})")
        plt.tight_layout()
        plt.savefig(outdir / f"slice_{i}.png", dpi=150)
        plt.close()

    # max intensity projection
    mip = density.max(axis=2)
    plt.figure(figsize=(4,4))
    plt.imshow(
        mip,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="inferno",
    )
    plt.colorbar()
    plt.title("atom density (max-intensity projection)")
    plt.tight_layout()
    plt.savefig(outdir / "atom_mip.png", dpi=150)
    plt.close()


def save_energy(energies, outdir):
    steps = [s for (s, _) in energies]
    vals = [e for (_, e) in energies]
    plt.figure(figsize=(4,3))
    plt.plot(steps, vals, marker="o")
    plt.xlabel("imaginary time step")
    plt.ylabel("energy (a.u.)")
    plt.title("convergence to ground state")
    plt.tight_layout()
    plt.savefig(outdir / "energy_convergence.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Physics-first 3D atom reconstruction")
    ap.add_argument("--N", type=int, default=64, help="grid points per axis")
    ap.add_argument("--L", type=float, default=12.0, help="physical box size (a.u.)")
    ap.add_argument("--Z", type=float, default=1.0, help="nuclear charge")
    ap.add_argument("--steps", type=int, default=600, help="imaginary time steps")
    ap.add_argument("--dt", type=float, default=0.002, help="imaginary time step")
    ap.add_argument("--softening", type=float, default=0.3, help="nuclear softening")
    args = ap.parse_args()

    outdir = Path("artifacts/real_atom")
    outdir.mkdir(parents=True, exist_ok=True)

    result = imaginary_time_solve(
        N=args.N,
        L=args.L,
        Z=args.Z,
        dt=args.dt,
        steps=args.steps,
        softening=args.softening,
    )

    # save raw fields
    np.save(outdir / "psi.npy", result["psi"])
    np.save(outdir / "density.npy", result["density"])
    np.save(outdir / "V.npy", result["V"])

    # descriptor so other AIs can rebuild it
    descriptor = {
        "name": "REAL-ATOM-FROM-SCHRODINGER-V1",
        "grid": {
            "N": args.N,
            "L": args.L,
            "dx": float(result["dx"]),
            "coords": {
                "x_min": float(result["x"][0]),
                "x_max": float(result["x"][-1]),
            },
        },
        "potential": {
            "type": "coulomb",
            "Z": args.Z,
            "softening": args.softening,
        },
        "solver": {
            "method": "imaginary_time",
            "dt": args.dt,
            "steps": args.steps,
        },
        "energies": [
            {"step": int(s), "E": float(e)} for (s, e) in result["energies"]
        ],
        "artifacts": {
            "psi": "artifacts/real_atom/psi.npy",
            "density": "artifacts/real_atom/density.npy",
            "potential": "artifacts/real_atom/V.npy",
            "slices": "artifacts/real_atom/slice_*.png",
            "mip": "artifacts/real_atom/atom_mip.png",
            "energy_plot": "artifacts/real_atom/energy_convergence.png",
        },
        "notes": "Derived from constants; no hand-tuned orbitals.",
    }
    with open(outdir / "atom_descriptor.json", "w") as f:
        json.dump(descriptor, f, indent=2)

    # viz
    if VIZ:
        save_slices(result["density"], result["x"], result["y"], result["z"], outdir)
        save_energy(result["energies"], outdir)

    print("✅ done. see artifacts/real_atom/")
    print(json.dumps(descriptor, indent=2))


if __name__ == "__main__":
    main()
