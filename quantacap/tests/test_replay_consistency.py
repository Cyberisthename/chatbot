from quantacap.core.adapter_store import create_adapter, load_adapter
from quantacap.quantum.circuits import Circuit
from quantacap.quantum.gates import H, RZ


def test_replay_matches_live():
    theta = 1.234
    circuit = Circuit(n=1, seed=424242)
    circuit.add(H(), [0])
    circuit.add(RZ(theta), [0])
    circuit.add(H(), [0])
    live_probs = circuit.probs().tolist()

    payload = {
        "state": {
            "n": 1,
            "probs": live_probs,
        },
        "meta": {"theta": theta},
    }
    create_adapter("test.demo.state", payload)
    rec = load_adapter("test.demo.state")

    saved = rec["data"]["state"]["probs"]
    assert len(saved) == len(live_probs)
    assert max(abs(a - b) for a, b in zip(saved, live_probs)) < 1e-10
