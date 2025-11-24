#!/usr/bin/env python3
"""Demo of Phase MLP classifier and RL Scientist."""

from __future__ import annotations

import random
from pathlib import Path

from jarvis5090x import (
    AdapterDevice,
    CentroidPhaseClassifier,
    DeviceKind,
    Jarvis5090X,
    MLPPhaseClassifier,
    OperationKind,
    PhaseDetector,
)


def demo_mlp_classifier() -> None:
    print("=" * 80)
    print("DEMO: MLP Phase Classifier vs Centroid Classifier")
    print("=" * 80)
    
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
    
    phase_types = [
        "ising_symmetry_breaking",
        "spt_cluster",
        "trivial_product",
        "pseudorandom",
    ]
    
    print("\n1. Generating training dataset (40 samples)...")
    for phase in phase_types:
        for i in range(10):
            detector.run_phase_experiment(
                phase_type=phase,
                system_size=32,
                depth=random.choice([6, 8, 10]),
                seed=random.randint(1, 10000),
                bias=random.choice([0.6, 0.7, 0.8]),
            )
    
    dataset = detector.build_dataset()
    print(f"   Generated {len(dataset)} training examples")
    
    train_dataset, test_dataset = dataset.split(0.8)
    print(f"   Split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    print("\n2. Training Centroid Classifier...")
    centroid = CentroidPhaseClassifier()
    centroid.train(train_dataset)
    centroid_results = centroid.evaluate(test_dataset)
    print(f"   Accuracy: {centroid_results['accuracy']:.4f}")
    
    print("\n3. Training MLP Classifier...")
    mlp = MLPPhaseClassifier()
    if not mlp.is_available():
        print("   ⚠ PyTorch not available - skipping MLP classifier")
        mlp_results = None
    else:
        mlp.train(train_dataset, epochs=50)
        mlp_results = mlp.evaluate(test_dataset)
        print(f"   Accuracy: {mlp_results['accuracy']:.4f}")
    
    print("\n4. Comparison:")
    print(f"   {'Classifier':<20} {'Accuracy':<12} {'Mean Confidence'}")
    print(f"   {'-' * 50}")
    print(f"   {'Centroid':<20} {centroid_results['accuracy']:<12.4f} {centroid_results['mean_confidence']:.4f}")
    if mlp_results:
        print(f"   {'MLP Neural Net':<20} {mlp_results['accuracy']:<12.4f} {mlp_results['mean_confidence']:.4f}")
        improvement = (mlp_results['accuracy'] - centroid_results['accuracy']) * 100
        if improvement > 0:
            print(f"\n   MLP improvement: +{improvement:.1f}% accuracy")
        else:
            print(f"\n   MLP performance: {improvement:.1f}% vs centroid")
    else:
        print(f"   {'MLP Neural Net':<20} (unavailable)")



def demo_rl_scientist() -> None:
    print("\n" + "=" * 80)
    print("DEMO: RL Lab Scientist (Q-Learning)")
    print("=" * 80)
    
    from experiments.rl_scientist import RLLabEnv, make_detector, train_q_agent
    
    print("\n1. Setting up lab environment...")
    detector = make_detector()
    env = RLLabEnv(detector)
    print(f"   Action space: {env.num_actions()} discrete actions")
    print(f"   Phases: {env.phase_types}")
    print(f"   Biases: {env.biases}")
    print(f"   Depths: {env.depths}")
    
    print("\n2. Training Q-learning agent (100 episodes)...")
    Q, history = train_q_agent(env, episodes=100)
    
    print("\n3. Results:")
    best_idx = max(range(len(Q)), key=lambda i: Q[i])
    phase, bias, depth = env.action_space[best_idx]
    
    print(f"   Best action discovered:")
    print(f"     Phase: {phase}")
    print(f"     Bias: {bias}")
    print(f"     Depth: {depth}")
    print(f"     Expected TRI: {Q[best_idx]:.4f}")
    
    avg_reward = sum(result.reward for result in history) / len(history)
    print(f"   Average reward: {avg_reward:.4f}")
    
    print("\n   Top 5 actions by Q-value:")
    top_indices = sorted(range(len(Q)), key=lambda i: Q[i], reverse=True)[:5]
    for rank, idx in enumerate(top_indices, 1):
        phase, bias, depth = env.action_space[idx]
        print(f"     {rank}. {phase[:20]:<20} bias={bias:.1f} depth={depth:2d} Q={Q[idx]:.4f}")


def main() -> None:
    random.seed(42)
    
    print("\n🧪 Jarvis-5090X: Phase MLP & RL Scientist Demo\n")
    
    demo_mlp_classifier()
    demo_rl_scientist()
    
    print("\n" + "=" * 80)
    print("Demo complete! See PHASE_MLP_RL_SCIENTIST_README.md for more details.")
    print("=" * 80)


if __name__ == "__main__":
    main()
