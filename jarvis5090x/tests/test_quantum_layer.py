from jarvis5090x.quantum_layer import Branch, QuantumApproximationLayer


def test_quantum_spawn():
    layer = QuantumApproximationLayer(max_branches=5)
    base_state = {"x": 0, "y": 0}
    variations = [{"x": 1}, {"x": 2}, {"x": 3}]

    branches = layer.spawn(base_state, variations)
    assert len(branches) == 4
    total_prob = sum(b.probability() for b in branches)
    assert 0.99 < total_prob < 1.01


def test_quantum_interfere():
    layer = QuantumApproximationLayer()
    base_state = {"value": 5}
    variations = [{"value": 10}, {"value": 15}]

    branches = layer.spawn(base_state, variations)
    scoring_fn = lambda state: state.get("value", 0)
    interfered = layer.interfere(branches, scoring_fn)

    assert len(interfered) > 0
    total_prob = sum(b.probability() for b in interfered)
    assert 0.99 < total_prob < 1.01


def test_quantum_collapse():
    layer = QuantumApproximationLayer(seed=42)
    base_state = {"score": 1}
    variations = [{"score": 10}, {"score": 20}]

    branches = layer.spawn(base_state, variations)
    collapsed = layer.collapse(branches, top_k=1)

    assert "score" in collapsed


def test_quantum_deterministic():
    layer1 = QuantumApproximationLayer(seed=123)
    layer2 = QuantumApproximationLayer(seed=123)

    base_state = {"value": 42}
    variations = [{"value": 100}]

    branches1 = layer1.spawn(base_state, variations)
    branches2 = layer2.spawn(base_state, variations)

    for b1, b2 in zip(branches1, branches2):
        assert abs(b1.amplitude - b2.amplitude) < 1e-9
