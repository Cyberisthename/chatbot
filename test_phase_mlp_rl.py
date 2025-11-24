#!/usr/bin/env python3
"""Basic tests for Phase MLP and RL components."""

from __future__ import annotations

import random
from pathlib import Path
import tempfile

from jarvis5090x import (
    AdapterDevice,
    CentroidPhaseClassifier,
    DeviceKind,
    Jarvis5090X,
    MLPPhaseClassifier,
    OperationKind,
    PhaseDataset,
    PhaseDetector,
    PhaseExample,
)


def test_mlp_classifier() -> None:
    print("Testing MLPPhaseClassifier...")
    
    mlp = MLPPhaseClassifier()
    if not mlp.is_available():
        print("  ⚠ PyTorch not available, skipping MLP tests")
        return
    
    # Create synthetic dataset
    dataset = PhaseDataset()
    for i in range(40):
        phase = ["phase_a", "phase_b", "phase_c"][i % 3]
        features = [float(i % 10), float(i % 5)] + [0.0] * 14
        dataset.add_example(
            PhaseExample(
                experiment_id=f"exp_{i}",
                phase_label=phase,
                feature_vector=features,
                params={},
            )
        )
    
    # Train and test
    train_dataset, test_dataset = dataset.split(0.75)
    
    result = mlp.train(train_dataset, epochs=10)
    assert "epochs" in result
    assert result["unique_labels"] == 3
    
    # Test prediction
    label, conf = mlp.predict(test_dataset.examples[0].feature_vector)
    assert label in ["phase_a", "phase_b", "phase_c"]
    assert 0.0 <= conf <= 1.0
    
    # Test evaluation
    eval_results = mlp.evaluate(test_dataset)
    assert "accuracy" in eval_results
    assert "confusion_matrix" in eval_results
    
    print(f"  ✓ Training: {result['training_examples']} examples")
    print(f"  ✓ Prediction: {label} (confidence: {conf:.4f})")
    print(f"  ✓ Evaluation: {eval_results['accuracy']:.4f} accuracy")


def test_dataset_io() -> None:
    print("Testing PhaseDataset I/O...")
    
    dataset = PhaseDataset()
    for i in range(10):
        dataset.add_example(
            PhaseExample(
                experiment_id=f"exp_{i}",
                phase_label=f"phase_{i % 3}",
                feature_vector=[float(i)] * 16,
                params={"seed": i},
            )
        )
    
    # Test save/load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    
    try:
        dataset.save_json(temp_path)
        loaded = PhaseDataset.load_json(temp_path)
        
        assert len(loaded) == len(dataset)
        assert loaded.examples[0].experiment_id == dataset.examples[0].experiment_id
        assert loaded.examples[0].phase_label == dataset.examples[0].phase_label
        
        print(f"  ✓ Saved and loaded {len(loaded)} examples")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_rl_env() -> None:
    print("Testing RLLabEnv...")
    
    from experiments.rl_scientist import RLLabEnv, make_detector
    
    detector = make_detector()
    env = RLLabEnv(detector)
    
    assert env.num_actions() == 4 * 3 * 3  # 4 phases × 3 biases × 3 depths
    
    # Test step
    state, reward, result = env.step(0)
    assert isinstance(reward, float)
    assert result.phase in env.phase_types
    assert result.bias in env.biases
    assert result.depth in env.depths
    
    print(f"  ✓ Action space: {env.num_actions()} actions")
    print(f"  ✓ Step executed: reward={reward:.4f}")


def test_integration() -> None:
    print("Testing full integration...")
    
    random.seed(42)
    
    devices = [
        AdapterDevice(
            id="quantum_0",
            label="Quantum Simulator",
            kind=DeviceKind.VIRTUAL,
            perf_score=50.0,
            max_concurrency=8,
            capabilities={OperationKind.QUANTUM},
        ),
    ]
    orchestrator = Jarvis5090X(devices)
    detector = PhaseDetector(orchestrator)
    
    # Generate small dataset
    phases = ["ising_symmetry_breaking", "spt_cluster"]
    for phase in phases:
        for _ in range(5):
            detector.run_phase_experiment(
                phase_type=phase,
                system_size=16,
                depth=4,
                seed=random.randint(1, 1000),
            )
    
    dataset = detector.build_dataset()
    assert len(dataset) == 10
    
    # Test both classifiers
    centroid = detector.centroid_classifier
    centroid.train(dataset)
    
    mlp = detector.mlp_classifier
    mlp_label = None
    if mlp is not None and mlp.is_available():
        mlp.train(dataset, epochs=10)
    
    # Predict on first example
    fv = dataset.examples[0].feature_vector
    c_label, c_conf = centroid.predict(fv)
    
    if mlp is not None and mlp.is_available():
        m_label, m_conf = mlp.predict(fv)
        mlp_label = m_label
    
    assert c_label in phases
    if mlp_label is not None:
        assert mlp_label in phases
    
    print(f"  ✓ Generated {len(dataset)} examples")
    print(f"  ✓ Centroid prediction: {c_label}")
    if mlp_label is not None:
        print(f"  ✓ MLP prediction: {mlp_label}")


def main() -> None:
    print("\n🧪 Running Phase MLP & RL Tests\n")
    
    tests = [
        test_mlp_classifier,
        test_dataset_io,
        test_rl_env,
        test_integration,
    ]
    
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            raise
    
    print("\n✓ All tests passed!\n")


if __name__ == "__main__":
    main()
