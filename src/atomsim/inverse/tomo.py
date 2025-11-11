"""Synthetic tomography (forward projections and filtered back-projection)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..render import viz3d

SEED = 424242


def generate_projections(
    density: NDArray[np.floating],
    angles: Iterable[float],
    noise: float = 0.0,
) -> NDArray[np.floating]:
    """Generate line-integral projections (Radon transform of mid-plane slice)."""

    slice_xy = density[:, :, density.shape[2] // 2]
    projections = []
    rng = np.random.default_rng(SEED)

    for theta in angles:
        rotated = ndimage.rotate(slice_xy, math.degrees(theta), reshape=False, order=1, mode="nearest")
        proj = rotated.sum(axis=0)
        if noise > 0:
            scale = proj.max() + 1e-6
            noisy = proj + noise * rng.normal(size=proj.shape) * scale
            proj = np.clip(noisy, 0.0, None)
        projections.append(proj)

    return np.stack(projections, axis=0)


def _ramp_filter(length: int) -> NDArray[np.floating]:
    freq = np.fft.rfftfreq(length)
    return np.abs(freq)


def filtered_back_projection(
    sinogram: NDArray[np.floating],
    angles: Iterable[float],
) -> NDArray[np.floating]:
    """Filtered back-projection (2D)."""

    num_angles, num_detectors = sinogram.shape
    filter_kernel = _ramp_filter(num_detectors)

    proj_fft = np.fft.rfft(sinogram, axis=1)
    proj_fft *= filter_kernel[None, :]
    filtered = np.fft.irfft(proj_fft, axis=1)

    recon = np.zeros((num_detectors, num_detectors), dtype=np.float64)
    angles_rad = list(angles)
    for i, theta in enumerate(angles_rad):
        backproj = np.tile(filtered[i], (num_detectors, 1))
        rotated = ndimage.rotate(backproj, -math.degrees(theta), reshape=False, order=1, mode="nearest")
        recon += rotated

    recon *= math.pi / len(angles_rad)
    recon = np.clip(recon, 0.0, None)
    return recon


def compute_metrics(reference: NDArray[np.floating], reconstruction: NDArray[np.floating]) -> Dict[str, float]:
    """Return SSIM, PSNR, and L2 metrics."""

    ref = reference.astype(np.float64)
    rec = reconstruction.astype(np.float64)

    mu_ref = ref.mean()
    mu_rec = rec.mean()

    sigma_ref = ref.var()
    sigma_rec = rec.var()
    covariance = ((ref - mu_ref) * (rec - mu_rec)).mean()

    c1 = 1e-4
    c2 = 1e-4
    ssim = ((2 * mu_ref * mu_rec + c1) * (2 * covariance + c2)) / (
        (mu_ref ** 2 + mu_rec ** 2 + c1) * (sigma_ref + sigma_rec + c2)
    )

    mse = np.mean((ref - rec) ** 2)
    dynamic = ref.max() - ref.min()
    if dynamic == 0:
        dynamic = 1.0
    psnr = 20 * math.log10(dynamic) - 10 * math.log10(mse + 1e-12)

    l2 = float(np.linalg.norm(ref - rec))

    return {"ssim": float(ssim), "psnr": float(psnr), "l2": l2}


def run_tomography(
    density: NDArray[np.floating],
    angles: Iterable[float],
    noise: float,
    out_dir: Path,
) -> Dict[str, float]:
    """End-to-end tomography pipeline with artifact saving."""

    out_dir.mkdir(parents=True, exist_ok=False)

    angles = list(angles)
    sinogram = generate_projections(density, angles, noise=noise)
    recon2d = filtered_back_projection(sinogram, angles)

    metrics = compute_metrics(density[:, :, density.shape[2] // 2], recon2d)

    np.save(out_dir / "sinogram.npy", sinogram.astype(np.float32))
    np.save(out_dir / "reconstruction.npy", recon2d.astype(np.float32))

    viz3d.save_sinogram(sinogram, out_dir)
    viz3d.save_reconstruction_slice(density, recon2d, out_dir)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


__all__ = [
    "generate_projections",
    "filtered_back_projection",
    "compute_metrics",
    "run_tomography",
]
