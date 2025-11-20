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
