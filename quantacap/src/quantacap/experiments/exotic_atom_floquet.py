import json
import math
from pathlib import Path

import numpy as np

# optional GIF stuff
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    GIF_OK = True
except Exception:
    GIF_OK = False


# ---------- helpers: paulis, kron, ops ----------

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def single_op(N, site, op):
    ops = []
    for i in range(N):
        ops.append(op if i == site else IDENTITY)
    return kron_all(ops)


def zz_op(N, i, j):
    ops = []
    for k in range(N):
        if k == i or k == j:
            ops.append(SIGMA_Z)
        else:
            ops.append(IDENTITY)
    return kron_all(ops)


# ---------- Hamiltonian builders ----------

def build_longrange_H(N, J_nn=1.0, J_lr=0.5, alpha=1.5, seed=424242):
    """
    H = sum_nn J_nn Z_i Z_{i+1} + sum_{i<j} J_lr * rand / |i-j|^alpha * Z_i Z_j
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    # nearest-neighbor
    for i in range(N - 1):
        H += J_nn * zz_op(N, i, i + 1)

    # long-range random
    for i in range(N):
        for j in range(i + 1, N):
            dist = abs(i - j)
            coeff = J_lr * rng.uniform(-1.0, 1.0) / (dist ** alpha)
            H += coeff * zz_op(N, i, j)

    return H


def build_Hx(N):
    Hx = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        Hx += single_op(N, i, SIGMA_X)
    return Hx


# ---------- evolution + measurements ----------

def time_evolve(state, H, dt):
    # U = e^{-i H dt} via eigendecomp (ok for N<=8)
    vals, vecs = np.linalg.eigh(H)
    U = vecs @ np.diag(np.exp(-1j * vals * dt)) @ np.linalg.inv(vecs)
    return U @ state


def entanglement_entropy(state, N, cut=None):
    if cut is None:
        cut = N // 2
    dimA = 2 ** cut
    dimB = 2 ** (N - cut)
    psi = state.reshape((dimA, dimB))
    rhoA = psi @ psi.conj().T
    vals = np.linalg.eigvalsh(rhoA)
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log2(vals)))


def z_expectations(state, N):
    """⟨Z_i⟩ for every site."""
    exps = []
    for i in range(N):
        Zi = single_op(N, i, SIGMA_Z)
        val = state.conj().T @ (Zi @ state)
        exps.append(float(np.real(val)))
    return exps


# ---------- synthetic atom rendering ----------

def build_atom_density(z_exps, grid=64, sigma=0.35):
    """
    Map per-qubit Z expectations into a 2D "atom" density.
    - we place N qubits on a circle
    - each qubit drops a gaussian blob with weight = (1 - z)/2  (prob of |1>)
    """
    N = len(z_exps)
    xs = np.linspace(-1.0, 1.0, grid)
    ys = np.linspace(-1.0, 1.0, grid)
    X, Y = np.meshgrid(xs, ys)
    density = np.zeros_like(X, dtype=float)

    radii = 0.55  # circle radius
    for k, z in enumerate(z_exps):
        theta = 2 * math.pi * k / N
        cx = radii * math.cos(theta)
        cy = radii * math.sin(theta)
        # probability from z: +1 => mostly |0>, -1 => mostly |1>
        p1 = (1.0 - z) / 2.0
        blob = p1 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
        density += blob

    # normalize
    if density.max() > 0:
        density = density / density.max()
    return X, Y, density


def save_atom_png(X, Y, density, path="artifacts/exotic_atom_density.png"):
    plt.figure(figsize=(4, 4))
    plt.imshow(density, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
    plt.colorbar(label="density")
    plt.title("Synthetic Atom Density (from qubit probs)")
    plt.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_atom_gif(frames, path="artifacts/exotic_atom_evolution.gif"):
    if not GIF_OK:
        return
    imgs = []
    for (X, Y, density) in frames:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(density, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
        ax.set_title("Synthetic Atom Evolution")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        tmp_path = "_tmp_frame.png"
        fig.savefig(tmp_path, dpi=110)
        plt.close(fig)
        imgs.append(Image.open(tmp_path).convert("P"))
    if imgs:
        Path(path).parent.mkdir(exist_ok=True)
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=120,
            loop=0,
        )


# ---------- main experiment ----------

def run_exotic_atom_floquet(
    N=8,
    steps=80,
    dt=0.05,
    drive_amp=1.0,
    drive_freq=2.0,
    J_nn=1.0,
    J_lr=0.5,
    alpha=1.5,
    seed=424242,
    make_gif=True,
    out_json="artifacts/exotic_atom_floquet.json",
):
    Path("artifacts").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    H_static = build_longrange_H(N, J_nn=J_nn, J_lr=J_lr, alpha=alpha, seed=seed)
    Hx = build_Hx(N)

    # start in |000...0>
    state = np.zeros((2 ** N,), dtype=complex)
    state[0] = 1.0 + 0j

    entropies = []
    energies = []
    drive_vals = []
    frames = []

    for step in range(steps):
        t = step * dt
        drive = drive_amp * math.cos(drive_freq * t)
        H_t = H_static + drive * Hx

        # evolve
        state = time_evolve(state, H_t, dt)
        state = state / np.linalg.norm(state)

        # measure
        S = entanglement_entropy(state, N)
        E = float(np.real(state.conj().T @ (H_t @ state)))
        zexp = z_expectations(state, N)

        entropies.append(S)
        energies.append(E)
        drive_vals.append(drive)

        # render frame for atom
        X, Y, dens = build_atom_density(zexp, grid=72, sigma=0.30)
        if make_gif:
            frames.append((X, Y, dens))

    # save final atom
    save_atom_png(X, Y, dens, path="artifacts/exotic_atom_density.png")
    if make_gif:
        save_atom_gif(frames, path="artifacts/exotic_atom_evolution.gif")

    out = {
        "experiment": "exotic_atom_floquet",
        "params": {
            "N": N,
            "steps": steps,
            "dt": dt,
            "drive_amp": drive_amp,
            "drive_freq": drive_freq,
            "J_nn": J_nn,
            "J_lr": J_lr,
            "alpha": alpha,
            "seed": seed,
        },
        "results": {
            "entanglement": entropies,
            "energies": energies,
            "drive": drive_vals,
            "last_z_expectations": zexp,
            "artifacts": {
                "png": "artifacts/exotic_atom_density.png",
                "gif": "artifacts/exotic_atom_evolution.gif" if make_gif else None,
            },
        },
        "notes": "Floquet-driven, long-range random ZZ Hamiltonian rendered as a synthetic atom.",
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    res = run_exotic_atom_floquet()
    print(json.dumps(res, indent=2))
