import math

import numpy as np

from quantacap.quantum.circuits import Circuit
from quantacap.quantum.gates import H, RZ


def test_fringe_sin2_law():
    steps = 41
    thetas = np.linspace(0, 2 * math.pi, steps)
    p1 = []
    for theta in thetas:
        circuit = Circuit(n=1, seed=424242)
        circuit.add(H(), [0])
        circuit.add(RZ(theta), [0])
        circuit.add(H(), [0])
        probs = circuit.probs()
        p1.append(float(probs[1]))
    mse = sum((p - (math.sin(t / 2) ** 2)) ** 2 for t, p in zip(thetas, p1)) / steps
    assert mse < 1e-3
