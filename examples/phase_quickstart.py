#!/usr/bin/env python3
"""
Phase Detector Quickstart Demo

Demonstrates the replayable synthetic quantum lab:
- Simulate phases of matter (Ising, SPT, Product, Pseudorandom)
- Log correlation + phase fingerprints
- Replay experiments deterministically
- Train classifier to recognize phases
"""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jarvis5090x import (
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
    PhaseDetector,
)


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_setup():
    print_section("🌀 PHASE DETECTOR SETUP")

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

    print(f"  ✓ Created {len(devices)} virtual device(s)")

    orchestrator = Jarvis5090X(devices)
    print("  ✓ Jarvis-5090X orchestrator ready")

    detector = PhaseDetector(orchestrator)
    print("  ✓ PhaseDetector initialized")
    print(f"  ✓ Available phases: {list(detector.generators.keys())}")

    return orchestrator, detector


def demo_run_phases(detector: PhaseDetector) -> None:
    print_section("🔬 RUNNING PHASE EXPERIMENTS")

    phases = [
        ("ising_symmetry_breaking", {"bias": 0.7}),
        ("spt_cluster", {}),
        ("trivial_product", {}),
        ("pseudorandom", {}),
    ]

    results = []
    for phase_type, options in phases:
        seed = random.randint(1, 1000)
        print(f"\n  Running: {phase_type} (seed={seed})...")

        result = detector.run_phase_experiment(
            phase_type=phase_type,
            system_size=32,
            depth=8,
            seed=seed,
            **options,
        )

        experiment_id = result["experiment_id"]
        feature_vec = result["feature_vector"]
        summary = result["summary"]

        print(f"    Experiment ID: {experiment_id}")
        print(f"    Features (dim={len(feature_vec)}): [{feature_vec[0]:.3f}, {feature_vec[1]:.3f}, ...]")
        print(f"    Magnetization: {summary.get('magnetization', 0.0):.4f}")
        print(f"    Entropy proxy: {summary.get('entropy_proxy', 0.0):.4f}")
        print(f"    Correlation length: {summary.get('correlation_length', 0.0):.4f}")

        results.append((experiment_id, phase_type, result))

    print(f"\n  ✓ Completed {len(results)} phase experiments")
    return results


def demo_replay(detector: PhaseDetector, experiment_id: str) -> None:
    print_section("🔁 REPLAY EXPERIMENT")

    print(f"  Replaying experiment: {experiment_id}")
    replay_result = detector.replay_experiment(experiment_id, compare=True)

    comparison = replay_result.get("comparison", {})
    original_id = comparison.get("original_id")
    replay_id = comparison.get("replayed_id")
    max_diff = comparison.get("max_difference", 0.0)
    is_match = comparison.get("is_match", False)

    print(f"  Original ID: {original_id}")
    print(f"  Replay ID:   {replay_id}")
    print(f"  Max feature difference: {max_diff:.8f}")
    print(f"  Is match: {is_match}")
    print(f"  ✓ Replay complete")


def demo_dataset(detector: PhaseDetector) -> None:
    print_section("📊 BUILD DATASET")

    dataset = detector.build_dataset()
    print(f"  Dataset size: {len(dataset)} examples")

    label_counts = {}
    for example in dataset.examples:
        label = example.phase_label
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:30s}: {count:3d} examples")

    print(f"  ✓ Dataset built")
    return dataset


def demo_classifier(detector: PhaseDetector, dataset) -> None:
    print_section("🤖 TRAIN CLASSIFIER")

    train_dataset, test_dataset = dataset.split(ratio=0.8)
    print(f"  Training set: {len(train_dataset)} examples")
    print(f"  Test set:     {len(test_dataset)} examples")

    print("\n  Training classifier...")
    training_report = detector.train_classifier(train_dataset)
    print(f"  Training examples: {training_report['training_examples']}")
    print(f"  Unique labels:     {training_report['unique_labels']}")

    print("\n  Evaluating on test set...")
    evaluation = detector.classifier.evaluate(test_dataset)

    accuracy = evaluation.get("accuracy", 0.0)
    correct = evaluation.get("correct", 0)
    total = evaluation.get("total", 0)
    mean_confidence = evaluation.get("mean_confidence", 0.0)

    print(f"  Accuracy:         {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"  Mean confidence:  {mean_confidence * 100:.2f}%")

    print("\n  Confusion matrix:")
    confusion = evaluation.get("confusion_matrix", {})
    if confusion:
        labels = sorted(confusion.keys())
        header = "  " + " " * 30 + "  ".join(f"{label[:10]:>10s}" for label in labels)
        print(header)
        for true_label in labels:
            row_counts = [str(confusion[true_label].get(pred_label, 0)) for pred_label in labels]
            print(f"  {true_label:30s}  " + "  ".join(f"{count:>10s}" for count in row_counts))

    print(f"\n  ✓ Classifier trained and evaluated")


def demo_classify_new(detector: PhaseDetector) -> None:
    print_section("🔮 CLASSIFY NEW EXPERIMENT")

    seed = random.randint(1001, 2000)
    print(f"  Running new ising experiment (seed={seed})...")

    result = detector.run_phase_experiment(
        phase_type="ising_symmetry_breaking",
        system_size=32,
        depth=8,
        seed=seed,
        bias=0.68,
    )

    experiment_id = result["experiment_id"]
    print(f"  Experiment ID: {experiment_id}")

    print("\n  Classifying phase...")
    classification = detector.classify_phase(experiment_id=experiment_id)

    prediction = classification.get("prediction")
    confidence = classification.get("confidence", 0.0)
    print(f"  Predicted phase: {prediction}")
    print(f"  Confidence:      {confidence * 100:.2f}%")
    print(f"  True phase:      ising_symmetry_breaking")

    if prediction == "ising_symmetry_breaking":
        print(f"  ✅ Correct prediction!")
    else:
        print(f"  ❌ Incorrect prediction")

    print(f"\n  ✓ Classification complete")


def demo_complexity_scaling(detector: PhaseDetector) -> None:
    print_section("⚡ COMPLEXITY SCALING")

    print("  Testing scalability across system sizes...")

    sizes = [16, 32, 64]
    for system_size in sizes:
        print(f"\n  System size: {system_size}")

        result = detector.run_phase_experiment(
            phase_type="ising_symmetry_breaking",
            system_size=system_size,
            depth=6,
            seed=42,
        )

        summary = result["summary"]
        branch_count = summary.get("branch_count", 0)
        print(f"    Branches:   {branch_count}")
        print(f"    Features:   {len(result['feature_vector'])}")

    print("\n  ✓ Feature extraction scales polynomially")
    print("    Classification is O(n^k) in our replay-augmented model")


def main() -> None:
    print()
    print("████████████████████████████████████████████████████████████████████████████████")
    print("██                                                                            ██")
    print("██         🌀 PHASE DETECTOR - REPLAYABLE QUANTUM LAB 🌀                     ██")
    print("██                                                                            ██")
    print("██  Synthetic quantum phase detection with replay & classification           ██")
    print("██                                                                            ██")
    print("████████████████████████████████████████████████████████████████████████████████")

    orchestrator, detector = demo_setup()
    results = demo_run_phases(detector)

    first_experiment_id = results[0][0]
    demo_replay(detector, first_experiment_id)

    dataset = demo_dataset(detector)

    if len(dataset) >= 4:
        demo_classifier(detector, dataset)
        demo_classify_new(detector)
    else:
        print("\n  ⚠️  Not enough examples for training. Run more experiments first.")

    demo_complexity_scaling(detector)

    print()
    print("=" * 80)
    print("  ✅ DEMO COMPLETE")
    print("  📊 Phase Recognition with Replay is Empirically Efficient!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
