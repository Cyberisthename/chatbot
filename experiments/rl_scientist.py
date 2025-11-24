#!/usr/bin/env python3
"""Simple reinforcement learning agent that explores time-reversal experiments."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis5090x import AdapterDevice, DeviceKind, Jarvis5090X, OperationKind, PhaseDetector
from experiments.discovery_suite import run_time_reversal_test


@dataclass
class ExperimentResult:
    phase: str
    bias: float
    depth: int
    tri: float
    reward: float


class RLLabEnv:
    def __init__(self, detector: PhaseDetector):
        self.detector = detector
        self.phase_types = [
            "ising_symmetry_breaking",
            "spt_cluster",
            "trivial_product",
            "pseudorandom",
        ]
        self.biases = [0.6, 0.7, 0.8]
        self.depths = [4, 8, 12]

        self.action_space: List[Tuple[str, float, int]] = []
        for phase in self.phase_types:
            for bias in self.biases:
                for depth in self.depths:
                    self.action_space.append((phase, bias, depth))

    def num_actions(self) -> int:
        return len(self.action_space)

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, ExperimentResult]:
        phase, bias, depth = self.action_space[action_idx]
        tri_result = run_time_reversal_test(
            self.detector,
            phase_type=phase,
            system_size=32,
            depth=depth,
            bias=bias,
            seed=random.randint(1, 10_000),
        )
        reward = tri_result["TRI"]
        info = {
            "phase": phase,
            "bias": bias,
            "depth": depth,
            "TRI": reward,
        }
        state: Dict[str, Any] = {}
        result = ExperimentResult(phase=phase, bias=bias, depth=depth, tri=reward, reward=reward)
        return state, reward, result


def make_detector() -> PhaseDetector:
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
    return PhaseDetector(orchestrator)


def train_q_agent(env: RLLabEnv, episodes: int = 2000) -> Tuple[List[float], List[ExperimentResult]]:
    num_actions = env.num_actions()
    Q = [0.0 for _ in range(num_actions)]
    alpha = 0.1
    epsilon = 0.2

    history: List[ExperimentResult] = []

    for episode in range(episodes):
        if random.random() < epsilon:
            action = random.randrange(num_actions)
        else:
            max_q = max(Q)
            candidates = [idx for idx, value in enumerate(Q) if value == max_q]
            action = random.choice(candidates)

        _state, reward, result = env.step(action)
        Q[action] = (1 - alpha) * Q[action] + alpha * reward
        history.append(result)

        if (episode + 1) % 100 == 0:
            best_q = max(Q)
            print(f"Episode {episode + 1}: best_Q={best_q:.4f}")

    return Q, history


def summarize_results(Q: List[float], env: RLLabEnv) -> None:
    print("\n=== Final Q-values ===")
    for idx, value in enumerate(Q):
        phase, bias, depth = env.action_space[idx]
        print(f"Action {idx:2d}: phase={phase:25s} bias={bias:.2f} depth={depth:2d} Q={value:.4f}")

    best_idx = max(range(len(Q)), key=lambda i: Q[i])
    phase, bias, depth = env.action_space[best_idx]
    print("\nBest action found:")
    print(f"  phase={phase}")
    print(f"  bias={bias}")
    print(f"  depth={depth}")
    print(f"  expected_TRI={Q[best_idx]:.4f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a simple RL agent for the lab environment")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    detector = make_detector()
    env = RLLabEnv(detector)
    Q, history = train_q_agent(env, episodes=args.episodes)
    summarize_results(Q, env)

    avg_reward = sum(result.reward for result in history) / len(history) if history else 0.0
    print(f"\nAverage reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
