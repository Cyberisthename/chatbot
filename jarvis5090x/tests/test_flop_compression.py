from jarvis5090x.flop_compression import FlopCompressionLayer


def test_compression_layer_learns_and_compresses():
    layer = FlopCompressionLayer(max_bases=5, stability_threshold=2, tolerance=0.01)
    op_type = "matmul"
    signature = "basis-demo"

    base_payload = {"matrix": [1.0, 2.0, 3.0], "bias": 0.5}

    for _ in range(3):
        result = layer.maybe_compress(op_type, signature, base_payload)

    assert layer.has_basis(op_type, signature)
    compressed = layer.maybe_compress(op_type, signature, {"matrix": [1.0, 2.0, 3.0], "bias": 0.5})
    assert compressed.get("__compressed__") is True
    info = compressed["__compression__"]
    assert info["basis_key"]
    assert info["coefficients"]


def test_compression_reuses_basis_for_similar_payloads():
    layer = FlopCompressionLayer(max_bases=5, stability_threshold=2, tolerance=0.1)
    op_type = "vector_op"
    signature = "shared_basis"

    payload_a = {"values": [10.0, 20.0, 30.0], "scale": 1.0}
    layer.maybe_compress(op_type, signature, payload_a)
    layer.maybe_compress(op_type, signature, payload_a)
    compressed_a = layer.maybe_compress(op_type, signature, payload_a)
    assert compressed_a.get("__compressed__") is True

    payload_b = {"values": [10.1, 19.9, 30.05], "scale": 1.0}
    compressed_b = layer.maybe_compress(op_type, signature, payload_b)
    assert compressed_b.get("__compressed__") is True
    assert compressed_a["__compression__"]["basis_key"] == compressed_b["__compression__"]["basis_key"]


def test_compression_stats_structure():
    layer = FlopCompressionLayer()
    stats = layer.stats()
    assert set(stats.keys()) == {"basis_count", "observations", "stable_bases"}
