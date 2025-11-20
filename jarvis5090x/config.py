from dataclasses import dataclass


@dataclass
class Jarvis5090XConfig:
    compression_max_bases: int = 5000
    compression_stability_threshold: int = 4
    compression_tolerance: float = 1e-8
    cache_max_items: int = 200000
    quantum_max_branches: int = 64
    quantum_seed: int = 42
    adapter_scheduler_interval: float = 0.01
    benchmark_hook_enabled: bool = True


DEFAULT_CONFIG = Jarvis5090XConfig()


EXTREME_CONFIG = Jarvis5090XConfig(
    compression_max_bases=20000,
    compression_stability_threshold=3,
    compression_tolerance=1e-6,
    cache_max_items=1_000_000,
    quantum_max_branches=128,
    quantum_seed=42,
    adapter_scheduler_interval=0.005,
    benchmark_hook_enabled=True,
)
