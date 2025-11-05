import numpy as np

from atomsim.hydrogen import solver


def test_2s_has_radial_node():
    res = solver.solve_excited(n=2, l=0, m=0, N=32, L=10.0, Z=1.0, steps=200, dt=0.01, eps=0.3)
    line = res.psi[res.psi.shape[0] // 2, res.psi.shape[1] // 2, :].real
    non_zero = line[np.abs(line) > 1e-4]
    signs = np.sign(non_zero)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    assert sign_changes >= 1


def test_2p_has_angular_node():
    res = solver.solve_excited(n=2, l=1, m=0, N=32, L=10.0, Z=1.0, steps=200, dt=0.01, eps=0.3)
    plane = res.psi[:, :, res.psi.shape[2] // 2].real
    midpoint_row = plane[plane.shape[0] // 2, :]
    signs = np.sign(midpoint_row[np.abs(midpoint_row) > 1e-4])
    sign_changes = np.sum(signs[:-1] != signs[1:])
    assert sign_changes >= 1
