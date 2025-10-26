import numpy as np

from quantacap.experiments.atom1d import generate_atom_state


def test_atom1d_density_normalized_and_symmetric():
    grid, psi, density = generate_atom_state(n=3, L=3.0, sigma=0.7)
    assert abs(density.sum() - 1.0) < 1e-6
    assert np.allclose(density, density[::-1], atol=1e-6)
    assert psi.shape == (8, 1)
