# quantacap/experiments/qcr_atom_reconstruct.py
#
# QCR = Quantum Coordinate Reconstruction
# Goal: given qubit-like probabilities (or generate synthetic ones),
#       reconstruct a 3D atomic density in a cube, track convergence over time,
#       and emit a reproducible "atom constant" JSON so other AIs can rebuild it.

import json
import math
import argparse
from pathlib import Path

import numpy as np

# optional viz
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    VIZ_OK = True
except Exception:
    VIZ_OK = False


def make_grid(R=1.0, N=64):
    xs = np.linspace(-R, R, N)
    ys = np.linspace(-R, R, N)
    zs = np.linspace(-R, R, N)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


def seed_qubit_weights(n_qubits=8, seed=424242):
    """
    If we don't have real ⟨Z⟩ from your quantum run, we make some
    synthetic-but-plausible ones. You can replace this with real data.
    """
    rng = np.random.default_rng(seed)
    # values in [-1, 1]
    zexp = rng.uniform(-1.0, 1.0, size=(n_qubits,))
    return zexp.tolist()


def density_from_qubits(zexp, X, Y, Z):
    """
    Turn per-qubit expectations into 3D density.
    This is very close to what we talked about before, but without assuming
    a fixed orbital. Each qubit becomes a lobe at a different angle.
    """
    Nq = len(zexp)
    density = np.zeros_like(X, dtype=float)

    # nucleus (always present)
    nucleus = np.exp(-(X**2 + Y**2 + Z**2) / (2 * (0.16**2)))
    density += 0.35 * nucleus

    for k, z in enumerate(zexp):
        # turn z in [-1,1] into weight in [0,1]
        w = (1.0 - z) / 2.0
        if w < 1e-4:
            continue

        theta = 2 * math.pi * (k / max(1, Nq))
        phi = 0.6 + 1.2 * ((k % 4) / 4.0)  # spread vertically

        cx = 0.6 * math.cos(theta) * math.sin(phi)
        cy = 0.6 * math.sin(theta) * math.sin(phi)
        cz = 0.6 * math.cos(phi)

        sigma = 0.22
        blob = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2))
        density += w * blob

    # normalize 0..1
    mx = density.max()
    if mx > 0:
        density = density / mx
    return density


def qcr_iterate(
    X,
    Y,
    Z,
    zexp,
    iters=80,
    smooth=0.15,
    conv_tol=1e-4,
):
    """
    Main QCR loop.
    At every step:
      - recompute density from qubits
      - blend with previous (like "observation")
      - measure convergence
    Returns: final density, convergence history, frames
    """
    density = np.zeros_like(X, dtype=float)
    frames = []
    conv_hist = []

    prev = None
    for t in range(iters):
        base = density_from_qubits(zexp, X, Y, Z)

        if t == 0:
            density = base
        else:
            # exponential moving average = "measurement makes it real"
            density = (1 - smooth) * density + smooth * base

        # convergence measure: L2 diff to previous
        if prev is None:
            diff = 1.0
        else:
            diff = float(np.linalg.norm(density - prev) / (np.linalg.norm(prev) + 1e-9))
        conv_hist.append(diff)
        prev = density.copy()

        # keep some frames for GIF
        if VIZ_OK and (t % max(1, iters // 32) == 0 or t == iters - 1):
            frames.append(density[:, :, density.shape[2] // 2].copy())

        # early stop
        if diff < conv_tol:
            break

    return density, conv_hist, frames


def save_slices_3d(density, R=1.0, out_dir="artifacts/qcr"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    N = density.shape[0]
    z_ids = [int(N*0.15), int(N*0.35), int(N*0.5), int(N*0.7), int(N*0.9)]
    for i, zidx in enumerate(z_ids):
        plt.figure(figsize=(4, 4))
        plt.imshow(
            density[:, :, zidx],
            origin="lower",
            extent=[-R, R, -R, R],
            cmap="viridis",
        )
        plt.colorbar()
        plt.title(f"QCR atom slice z={zidx}/{N}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/slice_{i}.png", dpi=160)
        plt.close()

    # max-intensity projection
    mip = density.max(axis=2)
    plt.figure(figsize=(4, 4))
    plt.imshow(mip, origin="lower", extent=[-R, R, -R, R], cmap="viridis")
    plt.colorbar()
    plt.title("QCR atom: max-intensity projection")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/atom_mip.png", dpi=160)
    plt.close()


def save_conv_plot(conv_hist, out_dir="artifacts/qcr"):
    if not VIZ_OK:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(conv_hist)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("relative change (log)")
    plt.title("QCR convergence")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/convergence.png", dpi=160)
    plt.close()


def save_flythrough(density, R=1.0, out_path="artifacts/qcr/atom_fly.gif"):
    if not VIZ_OK:
        return
    Path("artifacts/qcr").mkdir(parents=True, exist_ok=True)
    N = density.shape[0]
    frames = []
    for zidx in range(0, N, max(1, N // 36)):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(
            density[:, :, zidx],
            origin="lower",
            extent=[-R, R, -R, R],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        tmp = "_tmp_qcr_slice.png"
        fig.savefig(tmp, dpi=120)
        plt.close(fig)
        frames.append(Image.open(tmp).convert("P"))
    if frames:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )


def save_isosurface_mask(density, iso=0.35, out_path="artifacts/qcr/atom_isomask.npy"):
    """
    This doesn't do marching cubes (needs extra deps),
    but it DOES save the 3D boolean mask of the "surface" so
    any other tool / AI can later reconstruct the actual mesh.
    """
    mask = density >= iso
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask)
    return mask


def save_constant_json(
    R,
    N,
    iso,
    conv_hist,
    zexp,
    out_path="artifacts/qcr/atom_constant.json",
):
    """
    This is the "consint" you were talking about:
    a single JSON that says:
      - what grid
      - what iso
      - what qubit weights
      - what convergence we saw
    ANY AI can rebuild the same atom from this.
    """
    payload = {
        "name": "QCR-ATOM-V1",
        "grid": {"R": R, "N": N},
        "isosurface": {"iso": iso, "file": "atom_isomask.npy"},
        "qubits": {"z_expectations": zexp, "n_qubits": len(zexp)},
        "convergence": {
            "n_steps": len(conv_hist),
            "final_delta": float(conv_hist[-1]) if conv_hist else None,
            "history": conv_hist[:200],  # don't blow up the file
        },
        "notes": "Coordinate-level atom reconstruction from synthetic quantum run.",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_qcr_atom(
    N=64,
    R=1.0,
    iters=100,
    iso=0.35,
    n_qubits=8,
    seed=424242,
):
    Path("artifacts/qcr").mkdir(parents=True, exist_ok=True)

    X, Y, Z = make_grid(R=R, N=N)

    # in real run: load this from latest quantum/floquet artifact
    zexp = seed_qubit_weights(n_qubits=n_qubits, seed=seed)

    density, conv_hist, frames = qcr_iterate(
        X,
        Y,
        Z,
        zexp,
        iters=iters,
        smooth=0.18,
        conv_tol=1e-4,
    )

    # save slices + plots
    save_slices_3d(density, R=R, out_dir="artifacts/qcr")
    save_conv_plot(conv_hist, out_dir="artifacts/qcr")
    save_flythrough(density, R=R, out_path="artifacts/qcr/atom_fly.gif")

    # save isosurface mask
    save_isosurface_mask(density, iso=iso, out_path="artifacts/qcr/atom_isomask.npy")

    # save raw density (so we can re-visualize)
    np.save("artifacts/qcr/atom_density.npy", density)

    # save constant
    const = save_constant_json(
        R,
        N,
        iso,
        conv_hist,
        zexp,
        out_path="artifacts/qcr/atom_constant.json",
    )

    # top-level summary
    summary = {
        "experiment": "qcr_atom_reconstruct",
        "artifacts": {
            "slices": "artifacts/qcr/slice_*.png",
            "mip": "artifacts/qcr/atom_mip.png",
            "gif": "artifacts/qcr/atom_fly.gif" if VIZ_OK else None,
            "density": "artifacts/qcr/atom_density.npy",
            "isomask": "artifacts/qcr/atom_isomask.npy",
            "constant": "artifacts/qcr/atom_constant.json",
            "convergence_plot": "artifacts/qcr/convergence.png" if VIZ_OK else None,
        },
        "constant_excerpt": const,
    }

    with open("artifacts/qcr/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def cli():
    p = argparse.ArgumentParser(description="QCR: Quantum Coordinate Reconstruction (atom)")
    p.add_argument("--N", type=int, default=64, help="grid size per axis")
    p.add_argument("--R", type=float, default=1.0, help="spatial radius")
    p.add_argument("--iters", type=int, default=100, help="reconstruction iterations")
    p.add_argument("--iso", type=float, default=0.35, help="isosurface threshold")
    p.add_argument("--n-qubits", type=int, default=8, help="synthetic qubit count")
    p.add_argument("--seed", type=int, default=424242)
    args = p.parse_args()
    out = run_qcr_atom(
        N=args.N,
        R=args.R,
        iters=args.iters,
        iso=args.iso,
        n_qubits=args.n_qubits,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cli()
