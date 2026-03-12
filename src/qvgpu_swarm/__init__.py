"""
Q-vGPU Swarm: Quantum-Assisted Virtual GPU Swarm

A revolutionary framework that enables training modern AI on old/deprecated GPUs
and even CPU-only hardware by combining:

1. TCL (Thought Compression Layer) for gradient/state compression
2. Quantum probabilistic/time-coercion math for efficient training
3. VRAM paging and memory virtualization
4. Swarm distribution across multiple devices

This framework tricks AI frameworks into thinking they're on a massive GPU
while actually running on a network of old hardware.
"""

from .v_gpu_abstraction import VirtualGPU, VRAMPagingManager, UnifiedMemorySpace, VRAMProfile
from .tcl_gradient_compression import TCLGradientCompressor, CompressedTensor
from .quantum_probabilistic_training import QuantumProbabilisticTrainer, SuperpositionState, TimeCoercionOptimizer
from .swarm_distribution import SwarmCoordinator, RingAllReduceMesh, DeviceSlice, DeviceInfo, DeviceRole, DeviceStatus
from .unified_dispatcher import Q_vGPU_Bridge, UnifiedDispatcher, BridgeConfig
from .device_registry import DeviceRegistry, HardwareCapability, auto_configure

__all__ = [
    # Pillar 1: vGPU Abstraction
    "VirtualGPU",
    "VRAMPagingManager",
    "UnifiedMemorySpace",
    "VRAMProfile",
    
    # Pillar 2: TCL Compression
    "TCLGradientCompressor",
    "CompressedTensor",
    
    # Pillar 3: Quantum Probabilistic Training
    "QuantumProbabilisticTrainer",
    "SuperpositionState",
    "TimeCoercionOptimizer",
    
    # Pillar 4: Swarm Distribution
    "SwarmCoordinator",
    "RingAllReduceMesh",
    "DeviceSlice",
    "DeviceInfo",
    "DeviceRole",
    "DeviceStatus",
    
    # Bridge Layer
    "Q_vGPU_Bridge",
    "UnifiedDispatcher",
    "BridgeConfig",
    
    # Device Management
    "DeviceRegistry",
    "HardwareCapability",
    "auto_configure",
]

__version__ = "1.0.0"
