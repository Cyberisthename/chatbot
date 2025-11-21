from __future__ import annotations

from .adapter_cluster import AdapterCluster
from .asic_utils import (
    analyze_mining_performance,
    bruteforce_threshold_bits,
    combine_hashrates_real,
    compute_scale_factor,
    estimate_daily_reward,
    estimate_time_to_crack,
    format_hashrate,
    format_time,
    is_bruteforce_feasible,
    run_mining_job,
    to_real_hashrate,
)
from .config import DEFAULT_CONFIG, EXTREME_CONFIG, Jarvis5090XConfig
from .flop_compression import FlopCompressionLayer
from .infinite_cache import InfiniteMemoryCache
from .orchestrator import Jarvis5090X
from .virtual_gpu import VirtualGPU
from .quantum_layer import Branch, QuantumApproximationLayer
from .types import AdapterDevice, DeviceKind, OperationKind, OperationRequest

__version__ = "1.0.0"

__all__ = [
    "Jarvis5090X",
    "VirtualGPU",
    "FlopCompressionLayer",
    "InfiniteMemoryCache",
    "QuantumApproximationLayer",
    "Branch",
    "AdapterCluster",
    "AdapterDevice",
    "DeviceKind",
    "OperationKind",
    "OperationRequest",
    "Jarvis5090XConfig",
    "DEFAULT_CONFIG",
    "EXTREME_CONFIG",
    "is_bruteforce_feasible",
    "bruteforce_threshold_bits",
    "run_mining_job",
    "compute_scale_factor",
    "to_real_hashrate",
    "combine_hashrates_real",
    "estimate_daily_reward",
    "estimate_time_to_crack",
    "format_hashrate",
    "format_time",
    "analyze_mining_performance",
]
