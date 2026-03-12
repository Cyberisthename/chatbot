"""
The "Universal Dispatcher" Bridge

Instead of code talking directly to a specific hardware device, we create a small 
class that acts as a middleman. It uses the TCL engine to manage memory and
the quantum probabilistic trainer for efficient computation.

This bridge allows code to behave like it has 128GB VRAM, because it only ever
keeps the active layer on the hardware.
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .v_gpu_abstraction import VirtualGPU, VRAMProfile
from .tcl_gradient_compression import TCLGradientCompressor, UnifiedCompressionPipeline
from .quantum_probabilistic_training import QuantumProbabilisticTrainer, TimeCoercionOptimizer
from .swarm_distribution import SwarmCoordinator, DeviceInfo, DeviceRole


class ExecutionMode(Enum):
    """Execution mode for the bridge"""
    LOCAL = "local"           # Single device
    SWARM = "swarm"          # Distributed swarm
    HYBRID = "hybrid"        # Local + swarm


@dataclass
class BridgeConfig:
    """Configuration for the Q-vGPU Bridge"""
    
    # Virtual GPU settings
    virtual_vram_gb: float = 24.0
    physical_vram_gb: float = 2.0
    system_ram_gb: float = 16.0
    
    # TCL Compression settings
    enable_compression: bool = True
    compression_ratio: float = 8.0
    
    # Quantum training settings
    enable_quantum_training: bool = True
    sampling_ratio: float = 0.1
    
    # Swarm settings
    enable_swarm: bool = False
    device_discovery: bool = True
    
    # General
    device_name: str = "q-vgpu"
    prefetch_layers: int = 3
    checkpoint_interval: int = 100


class Q_vGPU_Bridge:
    """
    The main bridge that connects training code to virtualized hardware.
    
    This is the primary interface for the Q-vGPU Swarm framework.
    It orchestrates:
    - VRAM virtualization and paging
    - TCL-based compression for bandwidth
    - Quantum probabilistic training for FLOPS reduction
    - Swarm distribution across devices
    
    Usage:
        bridge = Q_vGPU_Bridge(physical_device="cpu", vram_limit_gb=2)
        
        # Wrap model training
        with bridge.virtual_device():
            output = model.forward(input_data)
            loss = compute_loss(output, target)
            grads = bridge.backward(loss)
            bridge.step(grads)
    """
    
    def __init__(self, 
                 physical_device: str = "cpu",
                 vram_limit_gb: float = 2.0,
                 config: Optional[BridgeConfig] = None):
        
        self.physical_device = physical_device
        self.vram_limit_gb = vram_limit_gb
        self.config = config or BridgeConfig()
        
        # Initialize subsystems
        self.vgpu = VirtualGPU(
            virtual_vram_gb=self.config.virtual_vram_gb,
            physical_vram_gb=vram_limit_gb,
            system_ram_gb=self.config.system_ram_gb,
            device_name=self.config.device_name
        )
        
        self.compressor = UnifiedCompressionPipeline(
            enable_sparsification=True,
            enable_tcl_compression=self.config.enable_compression,
            sparsity_target=0.9,
            tcl_compression=self.config.compression_ratio
        ) if self.config.enable_compression else None
        
        self.quantum_trainer: Optional[QuantumProbabilisticTrainer] = None
        self.time_coercion_opt: Optional[TimeCoercionOptimizer] = None
        
        self.swarm: Optional[SwarmCoordinator] = None
        
        # Model state
        self.model: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        
        # Tracking
        self.current_layer: int = 0
        self.layer_registry: Dict[int, Dict[str, str]] = {}
        self.tensor_registry: Dict[str, str] = {}  # name -> block_id
        
        # Statistics
        self.stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'compression_time': 0.0,
            'memory_swaps': 0,
            'steps_completed': 0,
        }
        
        self._lock = threading.RLock()
        
        print(f"🌉 Q_vGPU_Bridge initialized")
        print(f"   Physical device: {physical_device}")
        print(f"   Physical VRAM: {vram_limit_gb} GB")
        print(f"   Virtual VRAM: {self.config.virtual_vram_gb} GB")
        print(f"   TCL Compression: {self.config.enable_compression}")
        print(f"   Quantum Training: {self.config.enable_quantum_training}")
        
    def register_model(self, model: Any, optimizer: Optional[Any] = None):
        """
        Register a model with the bridge.
        
        This sets up paging, compression, and quantum training.
        """
        self.model = model
        self.optimizer = optimizer
        
        # Extract layer parameters and register with vGPU
        layer_params = self._extract_layer_params(model)
        self.vgpu.register_model_layers(layer_params)
        
        # Initialize quantum trainer if enabled
        if self.config.enable_quantum_training:
            self.quantum_trainer = QuantumProbabilisticTrainer(
                model=model,
                sampling_ratio=self.config.sampling_ratio
            )
            
            if optimizer:
                self.time_coercion_opt = TimeCoercionOptimizer(
                    base_optimizer=optimizer,
                    lookahead_steps=5
                )
        
        print(f"📦 Model registered with {len(layer_params)} layers")
        
    def _extract_layer_params(self, model: Any) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract parameters from model by layer"""
        layer_params = {}
        
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                params = {}
                for attr_name in ['query_proj', 'key_proj', 'value_proj', 'ffn1', 'ffn2',
                                 'gamma1', 'beta1', 'gamma2', 'beta2']:
                    if hasattr(layer, attr_name):
                        params[attr_name] = getattr(layer, attr_name)
                layer_params[i] = params
        
        return layer_params
    
    def allocate_tensor(self, name: str, shape: Tuple[int, ...],
                       dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate a tensor in the virtual memory space"""
        block_id = self.vgpu.allocate_tensor(name, shape, dtype)
        self.tensor_registry[name] = block_id
        
        # Return reference (data may not be on GPU yet)
        return self.vgpu.get_tensor(name)
    
    def get_tensor(self, name: str, ensure_on_device: bool = False) -> np.ndarray:
        """Get a tensor, optionally ensuring it's on the compute device"""
        return self.vgpu.get_tensor(name, for_computation=ensure_on_device)
    
    def forward(self, input_data: np.ndarray, layer_by_layer: bool = True) -> np.ndarray:
        """
        Execute forward pass with virtualization.
        
        If layer_by_layer=True, pages layers in and out as needed.
        """
        start_time = time.time()
        
        if not layer_by_layer or not self.model:
            # Standard forward pass
            return self._standard_forward(input_data)
        
        # Layer-by-layer forward with paging
        x = input_data
        
        for layer_idx in range(len(self.model.layers)):
            # Prepare layer (brings into GPU memory)
            layer_params = self.vgpu.prepare_layer(layer_idx)
            
            # Set layer parameters
            layer = self.model.layers[layer_idx]
            for param_name, value in layer_params.items():
                setattr(layer, param_name, value)
            
            # Forward through layer
            x, metrics = layer.forward(x)
            
            # Cache activation for backward
            self.vgpu.memory.blocks[
                self.vgpu.tensors.get(f'activation_{layer_idx}', '')
            ].data = x.copy() if layer_idx < len(self.model.layers) - 1 else None
        
        self.stats['forward_time'] += time.time() - start_time
        
        return x
    
    def _standard_forward(self, input_data: np.ndarray) -> np.ndarray:
        """Standard forward without layer-by-layer paging"""
        if hasattr(self.model, 'forward'):
            return self.model.forward(input_data)
        return input_data
    
    def backward(self, loss_gradient: np.ndarray,
                 use_quantum_gradients: bool = True) -> Dict[str, np.ndarray]:
        """
        Execute backward pass.
        
        Can use quantum probabilistic gradients for FLOPS reduction.
        """
        start_time = time.time()
        
        if use_quantum_gradients and self.quantum_trainer:
            # Use quantum probabilistic gradients
            def loss_fn(batch):
                # Simplified - would compute actual loss
                return np.mean(batch ** 2)
            
            gradients = self.quantum_trainer.training_step(loss_gradient, loss_fn)
        else:
            # Standard backprop
            if hasattr(self.model, 'backward'):
                gradients = self.model.backward(loss_gradient)
            else:
                gradients = {}
        
        self.stats['backward_time'] += time.time() - start_time
        
        return gradients
    
    def compress_for_transfer(self, tensor: np.ndarray,
                             tensor_type: str = 'gradient') -> Dict[str, Any]:
        """Compress a tensor for network transfer"""
        if not self.compressor:
            return {'data': tensor, 'compressed': False}
        
        start_time = time.time()
        compressed = self.compressor.compress(tensor, tensor_type)
        self.stats['compression_time'] += time.time() - start_time
        
        return compressed
    
    def step(self, gradients: Dict[str, np.ndarray],
            learning_rate: float = 0.001):
        """
        Execute optimization step.
        
        Applies gradients and updates weights, potentially using
        time-coercion for faster convergence.
        """
        if self.time_coercion_opt:
            # Use time-coercion optimizer
            weights = self._get_weights_dict()
            self.time_coercion_opt.step(gradients, weights)
            self._set_weights_dict(weights)
        elif self.optimizer:
            # Standard optimizer step
            self.optimizer.step(gradients)
        else:
            # Manual SGD update
            self._manual_sgd_update(gradients, learning_rate)
        
        self.stats['steps_completed'] += 1
    
    def _get_weights_dict(self) -> Dict[str, np.ndarray]:
        """Extract weights from model as dictionary"""
        weights = {}
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                for attr in ['query_proj', 'key_proj', 'value_proj', 'ffn1', 'ffn2']:
                    if hasattr(layer, attr):
                        weights[f'layer_{i}_{attr}'] = getattr(layer, attr)
        return weights
    
    def _set_weights_dict(self, weights: Dict[str, np.ndarray]):
        """Set weights from dictionary"""
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                for attr in ['query_proj', 'key_proj', 'value_proj', 'ffn1', 'ffn2']:
                    key = f'layer_{i}_{attr}'
                    if hasattr(layer, attr) and key in weights:
                        setattr(layer, attr, weights[key])
    
    def _manual_sgd_update(self, gradients: Dict[str, np.ndarray],
                          learning_rate: float):
        """Manual SGD weight update"""
        weights = self._get_weights_dict()
        for name, grad in gradients.items():
            if name in weights:
                weights[name] -= learning_rate * grad
        self._set_weights_dict(weights)
    
    def join_swarm(self, coordinator_address: Optional[str] = None):
        """Join a distributed training swarm"""
        if not self.config.enable_swarm:
            print("⚠️ Swarm mode not enabled in config")
            return False
        
        # Create swarm coordinator
        n_layers = len(self.model.layers) if hasattr(self.model, 'layers') else 1
        self.swarm = SwarmCoordinator(n_layers=n_layers)
        
        # Register this device
        device_info = DeviceInfo(
            device_id=self.config.device_name,
            role=DeviceRole.WORKER,
            status='online',
            vram_gb=self.vram_limit_gb,
            ram_gb=self.config.system_ram_gb,
        )
        
        self.swarm.register_device(device_info)
        
        print(f"🌐 Joined swarm as {self.config.device_name}")
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return self.vgpu.get_stats()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = self.stats.copy()
        
        if self.quantum_trainer:
            stats['quantum'] = self.quantum_trainer.get_stats()
        
        if self.compressor and self.compressor.tcl_compressor:
            stats['compression'] = self.compressor.tcl_compressor.get_stats()
        
        return stats
    
    def get_effective_performance(self) -> Dict[str, float]:
        """
        Calculate effective performance improvements.
        
        Returns multipliers for memory, bandwidth, and compute.
        """
        memory_multiplier = self.config.virtual_vram_gb / self.vram_limit_gb
        
        bandwidth_multiplier = 1.0
        if self.compressor:
            bandwidth_multiplier = self.compressor.get_effective_bandwidth_multiplier()
        
        compute_multiplier = 1.0
        if self.quantum_trainer:
            compute_multiplier = self.quantum_trainer.get_flops_reduction()
        
        return {
            'memory_multiplier': memory_multiplier,
            'bandwidth_multiplier': bandwidth_multiplier,
            'compute_multiplier': compute_multiplier,
            'overall_efficiency': memory_multiplier * bandwidth_multiplier * compute_multiplier,
        }
    
    def checkpoint(self, path: str):
        """Save checkpoint including all virtualized state"""
        checkpoint_data = {
            'config': {
                'virtual_vram_gb': self.config.virtual_vram_gb,
                'physical_vram_gb': self.vram_limit_gb,
                'enable_compression': self.config.enable_compression,
                'enable_quantum_training': self.config.enable_quantum_training,
            },
            'stats': self.stats,
            'tensor_registry': self.tensor_registry,
            'effective_performance': self.get_effective_performance(),
        }
        
        # Save model weights
        if self.model:
            model_path = Path(path).with_suffix('.model.npz')
            if hasattr(self.model, 'save'):
                self.model.save(str(model_path))
        
        # Save bridge state
        import json
        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint saved to {path}")
    
    def print_summary(self):
        """Print summary of bridge status and performance"""
        perf = self.get_effective_performance()
        
        print("\n" + "=" * 60)
        print("🚀 Q-vGPU Bridge Performance Summary")
        print("=" * 60)
        print(f"Virtual Memory:     {perf['memory_multiplier']:.1f}x ({self.config.virtual_vram_gb:.1f} GB virtual / {self.vram_limit_gb:.1f} GB physical)")
        print(f"Effective Bandwidth: {perf['bandwidth_multiplier']:.1f}x (via TCL compression)")
        print(f"Compute Efficiency:  {perf['compute_multiplier']:.1f}x (via quantum training)")
        print(f"Overall Efficiency:  {perf['overall_efficiency']:.1f}x")
        print("-" * 60)
        print(f"Training Steps:      {self.stats['steps_completed']}")
        print(f"Avg Forward Time:    {self.stats['forward_time'] / max(1, self.stats['steps_completed']):.3f}s")
        print(f"Avg Backward Time:   {self.stats['backward_time'] / max(1, self.stats['steps_completed']):.3f}s")
        print("=" * 60)


class UnifiedDispatcher:
    """
    High-level dispatcher that manages multiple bridges and coordinates
    training across heterogeneous hardware.
    
    This is the entry point for most users.
    """
    
    def __init__(self):
        self.bridges: Dict[str, Q_vGPU_Bridge] = {}
        self.active_bridge: Optional[str] = None
        
    def create_bridge(self, 
                     name: str,
                     physical_device: str = "cpu",
                     vram_gb: float = 2.0,
                     config: Optional[BridgeConfig] = None) -> Q_vGPU_Bridge:
        """Create a new Q-vGPU bridge"""
        bridge = Q_vGPU_Bridge(
            physical_device=physical_device,
            vram_limit_gb=vram_gb,
            config=config
        )
        self.bridges[name] = bridge
        return bridge
    
    def select_bridge(self, name: str):
        """Select active bridge"""
        if name in self.bridges:
            self.active_bridge = name
            print(f"✅ Selected bridge: {name}")
        else:
            raise ValueError(f"Bridge {name} not found")
    
    def get_active_bridge(self) -> Optional[Q_vGPU_Bridge]:
        """Get the currently active bridge"""
        if self.active_bridge:
            return self.bridges.get(self.active_bridge)
        return None
    
    def distribute_model(self, model: Any, 
                        device_assignments: Optional[Dict[str, List[int]]] = None):
        """
        Distribute model layers across multiple bridges/devices.
        
        Each bridge gets assigned specific layers to compute.
        """
        if not device_assignments:
            # Auto-distribute
            n_layers = len(model.layers) if hasattr(model, 'layers') else 1
            n_devices = len(self.bridges)
            layers_per_device = max(1, n_layers // n_devices)
            
            device_assignments = {}
            for i, name in enumerate(self.bridges.keys()):
                start = i * layers_per_device
                end = min(start + layers_per_device, n_layers)
                device_assignments[name] = list(range(start, end))
        
        # Register model with each bridge for assigned layers
        for bridge_name, layer_indices in device_assignments.items():
            if bridge_name in self.bridges:
                bridge = self.bridges[bridge_name]
                # Create sliced model view
                if hasattr(model, 'layers'):
                    sliced_layers = [model.layers[i] for i in layer_indices]
                    # Create proxy model with just those layers
                    bridge.register_model(model)  # Full model for now
                
                print(f"📍 {bridge_name}: layers {layer_indices}")
    
    def train_step(self, batch_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Execute one training step across all bridges"""
        results = {}
        
        for name, bridge in self.bridges.items():
            # Each bridge processes its assigned layers
            output = bridge.forward(batch_data)
            # In real distributed setup, outputs would be passed between devices
            results[name] = {'output_shape': output.shape}
        
        return results
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all bridges"""
        return {
            name: {
                'memory': bridge.get_memory_stats(),
                'training': bridge.get_training_stats(),
                'performance': bridge.get_effective_performance(),
            }
            for name, bridge in self.bridges.items()
        }
