import numpy as np

from quantacap.quantum.backend import create_circuit
from quantacap.quantum.gates import CNOT, H


def test_mps_matches_statevector_small():
    n = 4
    sv = create_circuit(n, backend="statevector", seed=424242)
    mps = create_circuit(n, backend="mps", seed=424242, chi=32)
    xp_sv = getattr(sv, "xp", None)
    xp_mps = getattr(mps, "xp", None)

    for q in range(n):
        sv.add(H(xp=xp_sv), [q])
        mps.add(H(xp=xp_mps), [q])
    for q in range(n - 1):
        sv.add(CNOT(xp=xp_sv), [q, q + 1])
        mps.add(CNOT(xp=xp_mps), [q, q + 1])

    probs_sv = sv.probs()
    probs_mps = mps.probs()
    mae = float(np.mean(np.abs(probs_sv - probs_mps)))
    assert mae < 1e-3
