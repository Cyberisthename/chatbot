import numpy as np
import pytest

from quantacap.experiments.atom1d import generate_atom_state, run_atom2d_transition


def test_atom1d_density_normalized_and_symmetric():
    grid, psi, density = generate_atom_state(n=3, L=3.0, sigma=0.7)
    assert abs(density.sum() - 1.0) < 1e-6
    assert np.allclose(density, density[::-1], atol=1e-6)
    assert psi.shape == (8, 1)


def test_atom2d_transition_outputs_density():
    result = run_atom2d_transition(
        n=4,
        L=4.0,
        sigma_primary=0.8,
        sigma_secondary=1.2,
        separation=1.0,
        adapter_id="atom2d.test",
        pi_adapter_id=None,
    )
    density = np.array(result["density"], dtype=float)
    assert density.shape == (1 << 2, 1 << 2)
    assert density.sum() == pytest.approx(1.0, rel=1e-6)
