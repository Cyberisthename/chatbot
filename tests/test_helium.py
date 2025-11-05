from atomsim.helium import solver


def test_helium_total_energy():
    res = solver.solve_helium(N=32, L=8.0, steps=150, dt=0.01, eps=0.3, mix=0.5)

    h_ground_single = -0.5
    assert res.total_energy < 2 * h_ground_single

    known_hf_bound = -2.85
    assert res.total_energy > known_hf_bound
    assert res.ee_energy > 0
