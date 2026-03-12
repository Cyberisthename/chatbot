"""
Device Registry and Hardware Capability Detection

Automatically detects available hardware and configures optimal settings
for the Q-vGPU Swarm framework.
"""

import numpy as np
import subprocess
import platform
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json


class HardwareType(Enum):
    """Types of hardware that can be used for training"""
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    CPU = "cpu"
    APPLE_SILICON = "apple_silicon"
    UNKNOWN = "unknown"


@dataclass
class HardwareCapability:
    """Complete hardware capability specification"""
    
    device_id: str
    device_name: str
    hardware_type: HardwareType
    
    # Memory
    vram_gb: float = 0.0  # GPU memory
    ram_gb: float = 0.0   # System memory
    shared_memory: bool = False
    
    # Compute
    compute_score: float = 1.0  # Relative to baseline CPU
    compute_capability: str = "unknown"
    tensor_cores: bool = False
    fp16_supported: bool = False
    bf16_supported: bool = False
    
    # Network
    network_interfaces: List[str] = field(default_factory=list)
    max_bandwidth_mbps: float = 1000.0
    
    # Status
    available: bool = True
    temperature_c: Optional[float] = None
    utilization_percent: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Shared memory systems (Apple Silicon, integrated graphics)
        if self.vram_gb == 0 and self.ram_gb > 0:
            self.shared_memory = True
            self.vram_gb = self.ram_gb * 0.5  # Assume half can be used as VRAM


class DeviceRegistry:
    """
    Discovers and registers available hardware devices.
    
    Automatically detects GPUs, CPUs, and their capabilities to
    configure optimal training settings.
    """
    
    def __init__(self):
        self.devices: Dict[str, HardwareCapability] = {}
        self._detected = False
        
    def detect_all(self) -> Dict[str, HardwareCapability]:
        """Detect all available hardware"""
        if self._detected:
            return self.devices
        
        print("🔍 Detecting available hardware...")
        
        # Detect NVIDIA GPUs
        self._detect_nvidia_gpus()
        
        # Detect AMD GPUs
        self._detect_amd_gpus()
        
        # Detect Intel GPUs
        self._detect_intel_gpus()
        
        # Detect Apple Silicon
        self._detect_apple_silicon()
        
        # Always detect CPU
        self._detect_cpu()
        
        self._detected = True
        
        print(f"✅ Detected {len(self.devices)} device(s)")
        for device in self.devices.values():
            mem_type = "VRAM" if not device.shared_memory else "Shared"
            print(f"   {device.device_name}: {device.vram_gb:.1f}GB {mem_type}, "
                  f"{device.compute_score:.1f}x compute")
        
        return self.devices
    
    def _detect_nvidia_gpus(self):
        """Detect NVIDIA GPUs using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        name = parts[0]
                        mem_str = parts[1].replace('MiB', '').replace('MB', '').strip()
                        compute_cap = parts[2]
                        
                        try:
                            vram_mb = int(mem_str)
                            vram_gb = vram_mb / 1024
                        except ValueError:
                            vram_gb = 4.0  # Default assumption
                        
                        # Calculate compute score based on architecture
                        compute_score = self._nvidia_compute_score(name, compute_cap)
                        
                        device = HardwareCapability(
                            device_id=f"cuda:{i}",
                            device_name=name,
                            hardware_type=HardwareType.NVIDIA_GPU,
                            vram_gb=vram_gb,
                            ram_gb=psutil.virtual_memory().total / (1024**3),
                            compute_score=compute_score,
                            compute_capability=compute_cap,
                            tensor_cores=self._has_tensor_cores(compute_cap),
                            fp16_supported=True,
                        )
                        
                        self.devices[f"cuda:{i}"] = device
                        
        except (subprocess.SubprocessError, FileNotFoundError):
            pass  # nvidia-smi not available
    
    def _detect_amd_gpus(self):
        """Detect AMD GPUs"""
        # ROCm detection would go here
        # For now, check for common AMD GPU indicators
        try:
            # Check for ROCm
            result = subprocess.run(
                ['rocminfo'], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse rocminfo output
                # Simplified - would need proper parsing
                pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    def _detect_intel_gpus(self):
        """Detect Intel GPUs"""
        # Intel GPU detection
        try:
            result = subprocess.run(
                ['intel-gpu-top', '-L'], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                pass  # Parse output
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    def _detect_apple_silicon(self):
        """Detect Apple Silicon (M1/M2/M3)"""
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                # Get chip info
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True
                )
                chip_name = result.stdout.strip()
                
                # Get memory
                mem_result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True
                )
                ram_bytes = int(mem_result.stdout.strip())
                ram_gb = ram_bytes / (1024**3)
                
                # Determine compute score based on chip
                compute_score = self._apple_compute_score(chip_name)
                
                device = HardwareCapability(
                    device_id="mps:0",
                    device_name=chip_name,
                    hardware_type=HardwareType.APPLE_SILICON,
                    vram_gb=ram_gb * 0.5,  # Unified memory
                    ram_gb=ram_gb,
                    shared_memory=True,
                    compute_score=compute_score,
                    fp16_supported=True,
                )
                
                self.devices["mps:0"] = device
                
            except (subprocess.SubprocessError, ValueError):
                pass
    
    def _detect_cpu(self):
        """Detect CPU capabilities"""
        cpu_info = {
            'name': platform.processor() or 'Unknown CPU',
            'cores': psutil.cpu_count(logical=False) or 1,
            'threads': psutil.cpu_count(logical=True) or 1,
        }
        
        # Estimate compute score based on cores and architecture
        compute_score = self._cpu_compute_score(cpu_info)
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        device = HardwareCapability(
            device_id="cpu",
            device_name=cpu_info['name'],
            hardware_type=HardwareType.CPU,
            vram_gb=0.0,  # No dedicated VRAM
            ram_gb=ram_gb,
            shared_memory=True,
            compute_score=compute_score,
            fp16_supported=True,  # Most modern CPUs support FP16 via AVX
        )
        
        self.devices["cpu"] = device
    
    def _nvidia_compute_score(self, name: str, compute_cap: str) -> float:
        """Calculate relative compute score for NVIDIA GPU"""
        base_score = 1.0
        
        # Parse compute capability
        try:
            major, minor = map(int, compute_cap.split('.'))
            cap_value = major * 10 + minor
            
            # Score based on architecture generation
            if cap_value >= 90:  # Hopper (H100)
                base_score = 50.0
            elif cap_value >= 80:  # Ampere (A100, RTX 30xx)
                base_score = 30.0
            elif cap_value >= 70:  # Turing (RTX 20xx)
                base_score = 15.0
            elif cap_value >= 60:  # Pascal (GTX 10xx)
                base_score = 8.0
            elif cap_value >= 50:  # Maxwell (GTX 9xx)
                base_score = 4.0
            elif cap_value >= 30:  # Kepler and older
                base_score = 2.0
            else:
                base_score = 1.0
                
        except ValueError:
            pass
        
        # Adjust for specific models
        if 'RTX 4090' in name:
            base_score *= 1.5
        elif 'Titan' in name or 'A100' in name:
            base_score *= 1.3
        elif '1060' in name or '1050' in name or '1650' in name:
            base_score *= 0.7
        
        return base_score
    
    def _apple_compute_score(self, chip_name: str) -> float:
        """Calculate compute score for Apple Silicon"""
        if 'M3' in chip_name:
            if 'Max' in chip_name:
                return 25.0
            elif 'Pro' in chip_name:
                return 15.0
            else:
                return 10.0
        elif 'M2' in chip_name:
            if 'Max' in chip_name:
                return 20.0
            elif 'Pro' in chip_name:
                return 12.0
            elif 'Ultra' in chip_name:
                return 30.0
            else:
                return 8.0
        elif 'M1' in chip_name:
            if 'Max' in chip_name:
                return 15.0
            elif 'Pro' in chip_name:
                return 10.0
            elif 'Ultra' in chip_name:
                return 25.0
            else:
                return 5.0
        return 3.0
    
    def _cpu_compute_score(self, cpu_info: Dict) -> float:
        """Calculate compute score for CPU"""
        cores = cpu_info.get('cores', 1)
        threads = cpu_info.get('threads', 1)
        
        # Base score on thread count
        base_score = threads * 0.2
        
        # Adjust for modern architectures
        name = cpu_info.get('name', '').lower()
        if 'intel' in name:
            if 'xeon' in name or 'i9' in name:
                base_score *= 1.5
            elif 'i7' in name:
                base_score *= 1.2
        elif 'amd' in name or 'ryzen' in name:
            if 'threadripper' in name or 'epyc' in name:
                base_score *= 1.8
            elif 'ryzen 9' in name:
                base_score *= 1.4
            elif 'ryzen 7' in name:
                base_score *= 1.2
        elif 'apple' in name:
            base_score *= 1.3
        
        return max(1.0, base_score)
    
    def _has_tensor_cores(self, compute_cap: str) -> bool:
        """Check if GPU has Tensor Cores"""
        try:
            major, _ = map(int, compute_cap.split('.'))
            return major >= 7  # Volta and newer
        except ValueError:
            return False
    
    def get_best_device(self) -> Optional[str]:
        """Get the best available device ID"""
        if not self.devices:
            self.detect_all()
        
        # Prefer dedicated GPUs with most VRAM
        gpus = [d for d in self.devices.values() 
                if d.hardware_type in [HardwareType.NVIDIA_GPU, 
                                      HardwareType.AMD_GPU,
                                      HardwareType.APPLE_SILICON]
                and not d.shared_memory]
        
        if gpus:
            # Sort by compute score
            best = max(gpus, key=lambda d: d.compute_score)
            return best.device_id
        
        # Check Apple Silicon
        apple = [d for d in self.devices.values()
                if d.hardware_type == HardwareType.APPLE_SILICON]
        if apple:
            return apple[0].device_id
        
        # Fall back to CPU
        return "cpu"
    
    def get_devices_by_capability(self, min_vram_gb: float = 0,
                                  min_compute_score: float = 0) -> List[HardwareCapability]:
        """Get devices meeting minimum requirements"""
        return [
            d for d in self.devices.values()
            if d.vram_gb >= min_vram_gb and d.compute_score >= min_compute_score
        ]
    
    def create_swarm_config(self) -> Dict[str, Any]:
        """Create optimal swarm configuration for detected devices"""
        if not self.devices:
            self.detect_all()
        
        total_compute = sum(d.compute_score for d in self.devices.values())
        total_vram = sum(d.vram_gb for d in self.devices.values())
        
        config = {
            'devices': {
                device_id: {
                    'name': d.device_name,
                    'type': d.hardware_type.value,
                    'vram_gb': d.vram_gb,
                    'compute_score': d.compute_score,
                    'compute_share': d.compute_score / total_compute if total_compute > 0 else 0,
                }
                for device_id, d in self.devices.items()
            },
            'total_compute_score': total_compute,
            'total_vram_gb': total_vram,
            'recommended_config': {
                'virtual_vram_multiplier': 4.0,  # 4x virtualization
                'enable_swarm': len(self.devices) > 1,
                'enable_compression': True,
                'enable_quantum_training': total_compute < 10,  # Enable for weak hardware
            }
        }
        
        return config
    
    def save_config(self, path: str):
        """Save detected configuration to file"""
        config = self.create_swarm_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Device configuration saved to {path}")
    
    def print_summary(self):
        """Print summary of detected hardware"""
        if not self.devices:
            self.detect_all()
        
        print("\n" + "=" * 70)
        print("🖥️  Hardware Detection Summary")
        print("=" * 70)
        
        for device_id, device in sorted(self.devices.items()):
            print(f"\n{device_id}: {device.device_name}")
            print(f"   Type: {device.hardware_type.value}")
            print(f"   VRAM: {device.vram_gb:.1f} GB", end="")
            if device.shared_memory:
                print(" (shared)")
            else:
                print()
            print(f"   RAM:  {device.ram_gb:.1f} GB")
            print(f"   Compute: {device.compute_score:.1f}x")
            print(f"   Tensor Cores: {'Yes' if device.tensor_cores else 'No'}")
            print(f"   FP16: {'Yes' if device.fp16_supported else 'No'}")
        
        total_compute = sum(d.compute_score for d in self.devices.values())
        total_vram = sum(d.vram_gb for d in self.devices.values())
        
        print("\n" + "-" * 70)
        print(f"Total Compute: {total_compute:.1f}x")
        print(f"Total VRAM:    {total_vram:.1f} GB")
        print(f"Effective VRAM (with 4x virtualization): {total_vram * 4:.1f} GB")
        print("=" * 70)


# Convenience function for quick setup
def auto_configure() -> Tuple['Q_vGPU_Bridge', Dict[str, Any]]:
    """
    Automatically detect hardware and configure optimal Q-vGPU bridge.
    
    Returns:
        (bridge, config): Configured bridge and configuration dict
    """
    from .unified_dispatcher import Q_vGPU_Bridge, BridgeConfig
    
    # Detect hardware
    registry = DeviceRegistry()
    devices = registry.detect_all()
    
    # Get best device
    best_device = registry.get_best_device()
    device_info = devices.get(best_device)
    
    if not device_info:
        best_device = "cpu"
        device_info = devices.get("cpu")
    
    # Create optimal config
    config = BridgeConfig(
        virtual_vram_gb=device_info.vram_gb * 4,  # 4x virtualization
        physical_vram_gb=device_info.vram_gb,
        system_ram_gb=device_info.ram_gb * 0.7,  # Use 70% of RAM
        enable_compression=True,
        enable_quantum_training=device_info.compute_score < 10,
        device_name=device_info.device_id,
    )
    
    # Create bridge
    bridge = Q_vGPU_Bridge(
        physical_device=device_info.device_id,
        vram_limit_gb=device_info.vram_gb,
        config=config
    )
    
    setup_info = {
        'device': device_info,
        'config': config,
        'all_devices': list(devices.keys()),
        'registry': registry,
    }
    
    return bridge, setup_info
