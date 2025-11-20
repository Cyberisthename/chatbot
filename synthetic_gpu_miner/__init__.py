"""
Synthetic GPU Miner - Advanced Mining Scheduler with Precompute Load Balancer

A hybrid CPU+GPU mining architecture that breaks work into micro-tasks,
precomputes everything possible, and adaptively schedules across all available
compute resources to maximize effective hash rate.

Architecture:
1. Protocol Layer - handles pool/node communication (Stratum)
2. Hash Core Layer - implements mining algorithms (SHA-256) for GPU/CPU
3. Precomputation & Cache Layer - precomputes and caches hash midstates
4. Synthetic GPU Layer - intelligent scheduler with task queues
5. Control & Telemetry Layer - performance monitoring and auto-tuning
"""

__version__ = "1.0.0"
__author__ = "Synthetic GPU Team"

from .work_unit import WorkUnit, Batch, Device, DeviceType
from .scheduler import SyntheticGPUScheduler
from .device_manager import DeviceManager
from .hash_core import HashCore
from .precompute_cache import PrecomputeCache
from .protocol_layer import ProtocolLayer
from .telemetry import TelemetryController

__all__ = [
    'WorkUnit',
    'Batch',
    'Device',
    'DeviceType',
    'SyntheticGPUScheduler',
    'DeviceManager',
    'HashCore',
    'PrecomputeCache',
    'ProtocolLayer',
    'TelemetryController',
]
