import numpy as np
import pytest

from quantacap.experiments.solve_atom_from_constants import (
    make_grid,
    coulomb_potential,
    laplacian_3d,
    normalize,
    compute_energy,
    imaginary_time_solve,
)


def test_make_grid():
    x, y, z, X, Y, Z = make_grid(N=16, L=4.0)
    assert len(x) == 16
    assert x[0] == pytest.approx(-2.0, abs=1e-6)
    assert x[-1] == pytest.approx(2.0, abs=1e-6)
    assert X.shape == (16, 16, 16)
    assert Y.shape == (16, 16, 16)
    assert Z.shape == (16, 16, 16)


def test_coulomb_potential_shape():
    x, y, z, X, Y, Z = make_grid(N=16, L=4.0)
    V = coulomb_potential(X, Y, Z, Zcharge=1.0, softening=0.3)
    assert V.shape == (16, 16, 16)
    assert np.all(V < 0)
    center_idx = len(x) // 2
    assert V[center_idx, center_idx, center_idx] < V[0, 0, 0]


def test_laplacian_3d_preserves_shape():
    x, y, z, X, Y, Z = make_grid(N=16, L=4.0)
    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)
    dx = x[1] - x[0]
    lap = laplacian_3d(psi, dx)
    assert lap.shape == psi.shape


def test_normalize():
    x, y, z, X, Y, Z = make_grid(N=16, L=4.0)
    dx = x[1] - x[0]
    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)
    psi_norm, prob = normalize(psi, dx)
    assert psi_norm.shape == psi.shape
    prob_check = np.sum(np.abs(psi_norm)**2) * (dx ** 3)
    assert prob_check == pytest.approx(1.0, abs=1e-6)


def test_compute_energy_negative_for_bound_state():
    x, y, z, X, Y, Z = make_grid(N=16, L=4.0)
    dx = x[1] - x[0]
    V = coulomb_potential(X, Y, Z, Zcharge=1.0, softening=0.3)
    psi = np.exp(-(X**2 + Y**2 + Z**2) / 2.0)
    psi, _ = normalize(psi, dx)
    E = compute_energy(psi, V, dx)
    assert E < 0


def test_imaginary_time_solve_converges():
    result = imaginary_time_solve(N=16, L=6.0, Z=1.0, dt=0.002, steps=100, softening=0.3)
    assert "psi" in result
    assert "density" in result
    assert "V" in result
    assert "energies" in result
    assert result["psi"].shape == (16, 16, 16)
    assert result["density"].shape == (16, 16, 16)
    energies = [e for (_, e) in result["energies"]]
    assert energies[-1] < energies[0]
    assert energies[-1] < 0


def test_imaginary_time_solve_density_normalized():
    result = imaginary_time_solve(N=16, L=6.0, Z=1.0, dt=0.002, steps=100, softening=0.3)
    dx = result["dx"]
    density = result["density"]
    total_prob = np.sum(density) * (dx ** 3)
    assert total_prob == pytest.approx(1.0, abs=1e-5)


def test_imaginary_time_solve_spherically_symmetric():
    result = imaginary_time_solve(N=16, L=6.0, Z=1.0, dt=0.002, steps=100, softening=0.3)
    density = result["density"]
    N = density.shape[0]
    center = N // 2
    d1 = density[center + 2, center, center]
    d2 = density[center, center + 2, center]
    d3 = density[center, center, center + 2]
    assert d1 == pytest.approx(d2, rel=0.1)
    assert d2 == pytest.approx(d3, rel=0.1)
