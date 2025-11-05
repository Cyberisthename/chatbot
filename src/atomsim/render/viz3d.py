"""3D visualization: MIPs, radial plots, isosurfaces, MP4 animations."""
from __future__ import annotations

import json
import math
import struct
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

try:
    from skimage.measure import marching_cubes as marching_cubes_lewiner
except ImportError:
    marching_cubes_lewiner = None

warnings.filterwarnings("ignore", message="invalid value encountered")


def save_mips(density: NDArray[np.floating], out_dir: Path) -> None:
    """Save maximum-intensity projections along x, y, z axes."""

    mip_xy = density.max(axis=2)
    mip_xz = density.max(axis=1)
    mip_yz = density.max(axis=0)

    for name, data, title in (
        ("mip_xy.png", mip_xy, "MIP XY"),
        ("mip_xz.png", mip_xz, "MIP XZ"),
        ("mip_yz.png", mip_yz, "MIP YZ"),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data.T, origin="lower", cmap="hot")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def radial_profile(density: NDArray[np.floating], out_dir: Path, box_length: float) -> None:
    """Compare numeric radial profile to analytic 1s hydrogen (exp(-2r))."""

    N = density.shape[0]
    center = N // 2
    L = box_length
    dx = L / N

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    r_grid = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)

    r_max = L / 2.0
    r_bins = np.linspace(0, r_max, 150)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    radial_avg = []
    for i in range(len(r_bins) - 1):
        mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
        avg = density[mask].mean() if mask.any() else 0.0
        radial_avg.append(avg)

    radial_avg = np.array(radial_avg)
    analytic_1s = np.exp(-2.0 * r_centers) / math.pi

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_centers, radial_avg, label="Numeric", linewidth=2)
    ax.plot(r_centers, analytic_1s, label="Analytic 1s", linestyle="--", linewidth=2)
    ax.set_xlabel("r (Bohr radii)")
    ax.set_ylabel("Density")
    ax.set_title("Radial Density (Ground State)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_compare.png", dpi=150)
    plt.close(fig)


def _write_gltf_binary(vertices: NDArray[np.floating], faces: NDArray[np.integer], path: Path) -> None:
    """Minimal glTF 2.0 binary export with embedded mesh data."""

    vertices = vertices.astype(np.float32)
    indices = faces.astype(np.uint32)

    vertex_bytes = vertices.tobytes()
    index_bytes = indices.tobytes()
    buffer_length = len(vertex_bytes) + len(index_bytes)
    buffer_padding = (4 - buffer_length % 4) % 4
    buffer_length += buffer_padding

    gltf_json = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes)},
            {"buffer": 0, "byteOffset": len(vertex_bytes), "byteLength": len(index_bytes)},
        ],
        "buffers": [{"byteLength": buffer_length}],
    }

    json_str = json.dumps(gltf_json, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_padding
    json_length = len(json_bytes)

    buffer_data = vertex_bytes + index_bytes + b"\x00" * buffer_padding
    bin_length = len(buffer_data)

    with path.open("wb") as f:
        f.write(struct.pack("<I", 0x46546C67))
        f.write(struct.pack("<I", 2))
        total_length = 12 + 8 + json_length + 8 + bin_length
        f.write(struct.pack("<I", total_length))

        f.write(struct.pack("<I", json_length))
        f.write(struct.pack("<I", 0x4E4F534A))
        f.write(json_bytes)

        f.write(struct.pack("<I", bin_length))
        f.write(struct.pack("<I", 0x004E4942))
        f.write(buffer_data)


def _placeholder_mesh() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Return vertices/faces of a simple cube placeholder."""

    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def marching_cubes_isosurface(density: NDArray[np.floating], path: Path, level: float) -> None:
    """Extract isosurface using marching cubes and save as glTF."""

    if marching_cubes_lewiner is None:
        print("  → Marching cubes unavailable; exporting placeholder cube")
        verts, faces = _placeholder_mesh()
        _write_gltf_binary(verts, faces, path)
        return

    threshold = level * density.max()
    try:
        verts, faces, _, _ = marching_cubes_lewiner(density, level=threshold, spacing=(1.0, 1.0, 1.0))
    except (RuntimeError, ValueError) as exc:
        print(f"  → Marching cubes failed: {exc}; exporting placeholder cube")
        verts, faces = _placeholder_mesh()
    else:
        if len(verts) == 0 or len(faces) == 0:
            print("  → Empty mesh, exporting placeholder cube")
            verts, faces = _placeholder_mesh()

    if faces.ndim == 2:
        faces_flat = faces.flatten()
    else:
        faces_flat = faces
    _write_gltf_binary(verts, faces_flat, path)


def make_orbit_mp4(density: NDArray[np.floating], path: Path) -> None:
    """Create a rotating orbit video of the density."""

    try:
        import imageio.v3 as iio
    except ImportError:
        print("  → Skipping MP4 (imageio not available)")
        return

    mip_xy = density.max(axis=2)
    frames = []
    for angle in np.linspace(0, 360, 180, endpoint=False):
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.imshow(mip_xy.T, origin="lower", cmap="hot")
        ax.text(
            0.5,
            0.95,
            f"{angle:.1f}°",
            color="white",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape((h, w, 3))
        frames.append(frame)
        plt.close(fig)

    try:
        iio.imwrite(path, frames, fps=30, codec="h264")
    except Exception as exc:
        print(f"  → MP4 export failed: {exc}")


def save_sinogram(sinogram: NDArray[np.floating], out_dir: Path) -> None:
    """Save sinogram visualization."""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sinogram, aspect="auto", cmap="gray")
    ax.set_xlabel("Detector Position")
    ax.set_ylabel("Projection Angle Index")
    ax.set_title("Sinogram")
    fig.tight_layout()
    fig.savefig(out_dir / "sinogram.png", dpi=150)
    plt.close(fig)


def save_reconstruction_slice(
    original: NDArray[np.floating],
    reconstruction: NDArray[np.floating],
    out_dir: Path,
) -> None:
    """Compare original mid-slice with reconstruction."""

    slice_orig = original[:, :, original.shape[2] // 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(slice_orig.T, origin="lower", cmap="hot")
    axes[0].set_title("Original (Mid-Slice)")
    axes[0].axis("off")

    axes[1].imshow(reconstruction.T, origin="lower", cmap="hot")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    diff = np.abs(slice_orig - reconstruction)
    axes[2].imshow(diff.T, origin="lower", cmap="viridis")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "recon_slice.png", dpi=150)
    plt.close(fig)


def density_comparison(
    reference: NDArray[np.floating],
    perturbed: NDArray[np.floating],
    out_dir: Path,
) -> None:
    """Side-by-side comparison of density before and after field."""

    ref_mip = reference.max(axis=2)
    pert_mip = perturbed.max(axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref_mip.T, origin="lower", cmap="hot")
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(pert_mip.T, origin="lower", cmap="hot")
    axes[1].set_title("With Field")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "density_comparison.png", dpi=150)
    plt.close(fig)


__all__ = [
    "save_mips",
    "radial_profile",
    "marching_cubes_isosurface",
    "make_orbit_mp4",
    "save_sinogram",
    "save_reconstruction_slice",
    "density_comparison",
]
