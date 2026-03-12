"""
Pillar 1: The vGPU Abstraction Layer (Software-Defined GPU)

Old GPUs crash during AI training because they run out of VRAM instantly.
The Fix: Create an abstraction driver at the OS level. When PyTorch asks 
for 24GB of VRAM, the vGPU says "Sure, I have it." In reality, the vGPU 
maps 2GB to the physical old GPU, 16GB to system RAM, and 6GB to a fast NVMe SSD.

Execution: By intercepting tensor operations, the vGPU schedules data to 
cycle in and out of the old GPU's small memory exactly when the computation 
is needed (Paged Attention applied to the whole training pipeline).
"""

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import hashlib


class MemoryTier(Enum):
    """Memory hierarchy tiers from fastest (GPU) to slowest (Disk)"""
    GPU_VRAM = "gpu_vram"           # Fastest - physical GPU memory
    SYSTEM_RAM = "system_ram"       # Fast - CPU system memory
    NVME_SSD = "nvme_ssd"          # Medium - NVMe swap
    SATA_SSD = "sata_ssd"          # Slow - SATA SSD swap
    DISK = "disk"                   # Slowest - HDD swap


@dataclass
class MemoryBlock:
    """A block of memory in the virtualized space"""
    block_id: str
    size_bytes: int
    tier: MemoryTier
    data: Optional[np.ndarray] = None
    device_ptr: Optional[int] = None  # Physical GPU pointer if on GPU
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    dirty: bool = False  # Modified since last sync
    compression_ratio: float = 1.0
    
    def touch(self):
        """Update access metadata"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass 
class VRAMProfile:
    """Configuration for virtualizing GPU memory"""
    virtual_vram_gb: float = 24.0  # What we tell PyTorch we have
    physical_gpu_vram_gb: float = 2.0  # Actual GPU VRAM
    system_ram_budget_gb: float = 16.0  # System RAM to use
    nvme_swap_gb: float = 64.0  # NVMe swap space
    prefetch_window_size: int = 3  # How many layers to prefetch
    eviction_policy: str = "lru"  # LRU, LFU, or adaptive
    enable_compression: bool = True
    compression_threshold_kb: float = 1024  # Compress tensors > 1MB


class UnifiedMemorySpace:
    """
    Manages a unified address space across GPU VRAM, System RAM, and Disk.
    
    This creates the illusion of a single large memory pool while managing
    the actual placement and migration of data across storage tiers.
    """
    
    def __init__(self, profile: VRAMProfile, swap_dir: str = "./vgpu_swap"):
        self.profile = profile
        self.swap_dir = Path(swap_dir)
        self.swap_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory pools by tier
        self.blocks: Dict[str, MemoryBlock] = {}
        self.gpu_resident: set = set()
        self.ram_resident: set = set()
        self.disk_resident: set = set()
        
        # Statistics
        self.stats = {
            'gpu_hits': 0,
            'ram_hits': 0, 
            'disk_hits': 0,
            'migrations_gpu_to_ram': 0,
            'migrations_ram_to_gpu': 0,
            'migrations_to_disk': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
        }
        
        self._lock = threading.RLock()
        self._access_history: List[Tuple[str, float]] = []
        
        # Pre-allocate RAM pool
        self._ram_pool: Dict[str, np.ndarray] = {}
        self._ram_used_bytes = 0
        self._ram_budget_bytes = int(profile.system_ram_budget_gb * 1024 * 1024 * 1024)
        
        print(f"🎭 UnifiedMemorySpace initialized:")
        print(f"   Virtual VRAM: {profile.virtual_vram_gb:.1f} GB")
        print(f"   Physical GPU: {profile.physical_gpu_vram_gb:.1f} GB")
        print(f"   System RAM: {profile.system_ram_budget_gb:.1f} GB")
        print(f"   NVMe Swap: {profile.nvme_swap_gb:.1f} GB")
        
    def allocate(self, shape: Tuple[int, ...], dtype = np.float32,
                 name: str = "tensor") -> str:
        """
        Allocate a tensor in the unified memory space.
        Returns a block_id that can be used to access the data.
        """
        size_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        block_id = f"{name}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        with self._lock:
            # Decide initial placement based on size and available GPU memory
            tier = self._select_tier(size_bytes)
            
            block = MemoryBlock(
                block_id=block_id,
                size_bytes=size_bytes,
                tier=tier,
                data=np.zeros(shape, dtype=dtype) if tier != MemoryTier.GPU_VRAM else None
            )
            
            self.blocks[block_id] = block
            self._update_residency(block_id, tier)
            
            return block_id
    
    def _select_tier(self, size_bytes: int) -> MemoryTier:
        """Select appropriate memory tier for allocation"""
        gpu_used = sum(self.blocks[b].size_bytes for b in self.gpu_resident)
        gpu_budget = int(self.profile.physical_gpu_vram_gb * 1024 * 1024 * 1024)
        
        if gpu_used + size_bytes < gpu_budget * 0.9:
            return MemoryTier.GPU_VRAM
        elif self._ram_used_bytes + size_bytes < self._ram_budget_bytes * 0.9:
            return MemoryTier.SYSTEM_RAM
        else:
            return MemoryTier.NVME_SSD
    
    def _update_residency(self, block_id: str, tier: MemoryTier):
        """Update which residency set a block belongs to"""
        self.gpu_resident.discard(block_id)
        self.ram_resident.discard(block_id)
        self.disk_resident.discard(block_id)
        
        if tier == MemoryTier.GPU_VRAM:
            self.gpu_resident.add(block_id)
        elif tier in (MemoryTier.SYSTEM_RAM,):
            self.ram_resident.add(block_id)
        else:
            self.disk_resident.add(block_id)
        
        self.blocks[block_id].tier = tier
    
    def access(self, block_id: str, for_computation: bool = False) -> np.ndarray:
        """
        Access data by block_id. Handles migration if needed.
        If for_computation=True, ensures data is on GPU.
        """
        with self._lock:
            if block_id not in self.blocks:
                raise ValueError(f"Block {block_id} not found")
            
            block = self.blocks[block_id]
            block.touch()
            self._access_history.append((block_id, time.time()))
            
            # If computation required, ensure on GPU
            if for_computation and block.tier != MemoryTier.GPU_VRAM:
                self._migrate_to_gpu(block_id)
                self.stats['gpu_hits'] += 1
            elif block.tier == MemoryTier.GPU_VRAM:
                self.stats['gpu_hits'] += 1
            elif block.tier == MemoryTier.SYSTEM_RAM:
                self.stats['ram_hits'] += 1
            else:
                self.stats['disk_hits'] += 1
            
            # Return data (may be in RAM or simulated GPU)
            if block.data is not None:
                return block.data
            else:
                # Should be on GPU - return from RAM cache or load
                return self._load_from_swap(block_id)
    
    def _migrate_to_gpu(self, block_id: str):
        """Migrate a block to GPU VRAM, evicting if necessary"""
        block = self.blocks[block_id]
        
        # Check if we need to evict
        gpu_used = sum(self.blocks[b].size_bytes for b in self.gpu_resident)
        gpu_budget = int(self.profile.physical_gpu_vram_gb * 1024 * 1024 * 1024)
        
        while gpu_used + block.size_bytes > gpu_budget * 0.95:
            # Evict LRU block
            evicted = self._evict_lru_from_gpu()
            if not evicted:
                break
            gpu_used = sum(self.blocks[b].size_bytes for b in self.gpu_resident)
        
        # Load data if not in memory
        if block.data is None:
            block.data = self._load_from_swap(block_id)
        
        self._update_residency(block_id, MemoryTier.GPU_VRAM)
        self.stats['migrations_ram_to_gpu'] += 1
        
    def _evict_lru_from_gpu(self) -> bool:
        """Evict least-recently-used block from GPU to RAM"""
        if not self.gpu_resident:
            return False
        
        # Find LRU
        lru_block = None
        lru_time = float('inf')
        for bid in self.gpu_resident:
            block = self.blocks[bid]
            if block.last_accessed < lru_time:
                lru_time = block.last_accessed
                lru_block = bid
        
        if lru_block:
            block = self.blocks[lru_block]
            # Check if we need to move to disk instead
            if self._ram_used_bytes + block.size_bytes > self._ram_budget_bytes:
                self._spill_to_disk(lru_block)
            else:
                self._update_residency(lru_block, MemoryTier.SYSTEM_RAM)
            self.stats['migrations_gpu_to_ram'] += 1
            return True
        return False
    
    def _spill_to_disk(self, block_id: str):
        """Move block to disk swap"""
        block = self.blocks[block_id]
        swap_path = self.swap_dir / f"{block_id}.npy"
        
        if block.data is not None:
            np.save(swap_path, block.data)
            block.data = None  # Free RAM
        
        self._update_residency(block_id, MemoryTier.NVME_SSD)
        self.stats['migrations_to_disk'] += 1
    
    def _load_from_swap(self, block_id: str) -> np.ndarray:
        """Load data from disk swap"""
        swap_path = self.swap_dir / f"{block_id}.npy"
        if swap_path.exists():
            return np.load(swap_path)
        
        # Return zeros if not found (shouldn't happen)
        block = self.blocks[block_id]
        shape = (block.size_bytes // 4,)  # Assume float32
        return np.zeros(shape, dtype=np.float32)
    
    def prefetch(self, block_ids: List[str]):
        """Prefetch blocks to GPU for upcoming computation"""
        gpu_budget = int(self.profile.physical_gpu_vram_gb * 1024 * 1024 * 1024)
        gpu_used = sum(self.blocks[b].size_bytes for b in self.gpu_resident)
        
        prefetched = 0
        for bid in block_ids:
            if bid in self.gpu_resident:
                continue
            
            block = self.blocks[bid]
            if gpu_used + block.size_bytes > gpu_budget * 0.9:
                break
            
            if block.data is None:
                block.data = self._load_from_swap(bid)
            
            self._update_residency(bid, MemoryTier.GPU_VRAM)
            gpu_used += block.size_bytes
            prefetched += 1
        
        if prefetched > 0:
            self.stats['prefetch_hits'] += prefetched
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current memory usage summary"""
        gpu_used = sum(self.blocks[b].size_bytes for b in self.gpu_resident)
        ram_used = sum(self.blocks[b].size_bytes for b in self.ram_resident)
        disk_used = sum(self.blocks[b].size_bytes for b in self.disk_resident)
        
        return {
            'virtual_total_gb': self.profile.virtual_vram_gb,
            'gpu_resident_gb': gpu_used / 1e9,
            'ram_resident_gb': ram_used / 1e9,
            'disk_resident_gb': disk_used / 1e9,
            'total_allocated_gb': (gpu_used + ram_used + disk_used) / 1e9,
            'gpu_utilization': gpu_used / (self.profile.physical_gpu_vram_gb * 1e9),
            'ram_utilization': ram_used / self._ram_budget_bytes,
            'stats': self.stats.copy()
        }


class VRAMPagingManager:
    """
    Manages paged memory for transformer layers.
    
    Only keeps active layer on GPU, pages others to RAM/Disk.
    """
    
    def __init__(self, memory_space: UnifiedMemorySpace, 
                 layers_per_prefetch: int = 2):
        self.memory = memory_space
        self.layers_per_prefetch = layers_per_prefetch
        self.layer_blocks: Dict[int, Dict[str, str]] = {}  # layer_idx -> {param_name -> block_id}
        self.current_layer: int = -1
        
    def register_layer_params(self, layer_idx: int, params: Dict[str, np.ndarray]):
        """Register layer parameters with the paging manager"""
        self.layer_blocks[layer_idx] = {}
        
        for param_name, tensor in params.items():
            block_id = self.memory.allocate(
                tensor.shape, 
                tensor.dtype,
                name=f"layer{layer_idx}_{param_name}"
            )
            
            # Store initial data
            block = self.memory.blocks[block_id]
            block.data = tensor.copy()
            
            self.layer_blocks[layer_idx][param_name] = block_id
    
    def prepare_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Prepare a layer for computation by ensuring it's on GPU.
        Pages out other layers if necessary.
        """
        if layer_idx not in self.layer_blocks:
            raise ValueError(f"Layer {layer_idx} not registered")
        
        # Prefetch this layer and next few
        prefetch_list = []
        for i in range(layer_idx, min(layer_idx + self.layers_per_prefetch, 
                                      max(self.layer_blocks.keys()) + 1)):
            prefetch_list.extend(self.layer_blocks[i].values())
        
        self.memory.prefetch(prefetch_list)
        
        # Access all params for this layer
        result = {}
        for param_name, block_id in self.layer_blocks[layer_idx].items():
            result[param_name] = self.memory.access(block_id, for_computation=True)
        
        self.current_layer = layer_idx
        return result
    
    def write_back_gradients(self, layer_idx: int, gradients: Dict[str, np.ndarray]):
        """Write back computed gradients to the unified memory"""
        if layer_idx not in self.layer_blocks:
            return
        
        for param_name, grad in gradients.items():
            # Store gradient in new block
            block_id = self.memory.allocate(
                grad.shape, grad.dtype,
                name=f"grad_layer{layer_idx}_{param_name}"
            )
            block = self.memory.blocks[block_id]
            block.data = grad.copy()
            block.dirty = True


class VirtualGPU:
    """
    Main vGPU abstraction that presents a unified interface.
    
    This class tricks the training code into thinking it has a massive GPU
    while actually orchestrating data movement across multiple memory tiers.
    """
    
    def __init__(self, 
                 virtual_vram_gb: float = 24.0,
                 physical_vram_gb: float = 2.0,
                 system_ram_gb: float = 16.0,
                 device_name: str = "q-vgpu"):
        
        self.profile = VRAMProfile(
            virtual_vram_gb=virtual_vram_gb,
            physical_gpu_vram_gb=physical_vram_gb,
            system_ram_budget_gb=system_ram_gb
        )
        
        self.memory = UnifiedMemorySpace(self.profile)
        self.paging = VRAMPagingManager(self.memory)
        self.device_name = device_name
        
        # Active tensors
        self.tensors: Dict[str, str] = {}  # name -> block_id
        
        print(f"🚀 VirtualGPU '{device_name}' initialized")
        print(f"   Virtual VRAM: {virtual_vram_gb:.1f} GB")
        print(f"   Physical: {physical_vram_gb:.1f} GB GPU + {system_ram_gb:.1f} GB RAM")
        
    def allocate_tensor(self, name: str, shape: Tuple[int, ...], 
                        dtype: np.dtype = np.float32) -> str:
        """Allocate a named tensor"""
        block_id = self.memory.allocate(shape, dtype, name=name)
        self.tensors[name] = block_id
        return block_id
    
    def get_tensor(self, name: str, for_computation: bool = False) -> np.ndarray:
        """Get tensor data, optionally ensuring it's on GPU"""
        if name not in self.tensors:
            raise ValueError(f"Tensor '{name}' not found")
        return self.memory.access(self.tensors[name], for_computation)
    
    def set_tensor(self, name: str, data: np.ndarray):
        """Set tensor data"""
        if name not in self.tensors:
            self.allocate_tensor(name, data.shape, data.dtype)
        
        block = self.memory.blocks[self.tensors[name]]
        block.data = data.copy()
        block.dirty = True
    
    def register_model_layers(self, layer_params: Dict[int, Dict[str, np.ndarray]]):
        """Register all model layers for paging"""
        for layer_idx, params in layer_params.items():
            self.paging.register_layer_params(layer_idx, params)
    
    def prepare_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Prepare a layer for forward/backward pass"""
        return self.paging.prepare_layer(layer_idx)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vGPU statistics"""
        return {
            'device': self.device_name,
            'memory': self.memory.get_memory_summary(),
            'active_tensors': len(self.tensors),
            'registered_layers': len(self.paging.layer_blocks)
        }
