import numpy as np

from atomsim.inverse import tomo


def test_tomography_ssim():
    N = 64
    axis = np.linspace(-1.0, 1.0, N, endpoint=False)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    density = np.exp(-10 * (x ** 2 + y ** 2 + z ** 2))

    angles = np.linspace(0, np.pi, 60, endpoint=False)
    sinogram = tomo.generate_projections(density, angles, noise=0.0)
    recon = tomo.filtered_back_projection(sinogram, angles)

    orig_slice = density[:, :, density.shape[2] // 2]
    metrics = tomo.compute_metrics(orig_slice, recon)

    assert metrics["ssim"] > 0.85
