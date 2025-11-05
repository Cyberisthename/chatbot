import numpy as np

from atomsim.hydrogen import solver


def test_ground_state_energy_and_norm():
    res = solver.solve_ground(N=48, L=10.0, Z=1.0, steps=250, dt=0.01, eps=0.3, record_every=10)
    norm = np.linalg.norm(res.psi)
    assert abs(norm - 1.0) < 1e-6

    energies = res.history.energies
    assert energies[0] >= energies[-1]
    assert -0.6 < res.energy < 0.0
