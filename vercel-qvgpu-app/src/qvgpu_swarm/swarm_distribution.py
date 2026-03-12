"""
Pillar 4: Swarm Distribution

A single old GPU is weak, but what if you have 4 old laptops and an old gaming rig?

The vGPU framework acts as a network mesh using Ring-AllReduce architecture
(similar to how supercomputers link GPUs).

Using TCL compression, network latency isn't a bottleneck anymore. Each old device
is assigned a "slice" of the AI model. Device A calculates layer 1, compresses it
via TCL, sends it over WiFi/Ethernet to Device B, which computes layer 2.
"""

import numpy as np
import threading
import queue
import time
import json
import socket
import pickle
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib


class DeviceRole(Enum):
    """Role of a device in the swarm"""
    COORDINATOR = "coordinator"  # Manages the swarm
    WORKER = "worker"            # Does computation
    HYBRID = "hybrid"           # Both coordinates and works


class DeviceStatus(Enum):
    """Status of a device in the swarm"""
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class DeviceInfo:
    """Information about a device in the swarm"""
    device_id: str
    role: DeviceRole
    status: DeviceStatus
    
    # Hardware capabilities
    vram_gb: float = 2.0
    ram_gb: float = 8.0
    compute_score: float = 1.0  # Relative compute capability
    network_mbps: float = 100.0
    
    # Current load
    vram_used_gb: float = 0.0
    cpu_percent: float = 0.0
    
    # Assignment
    assigned_layers: List[int] = field(default_factory=list)
    
    # Network info
    ip_address: Optional[str] = None
    port: int = 0
    
    # Metadata
    last_heartbeat: float = field(default_factory=time.time)
    total_flops_computed: float = 0.0


@dataclass
class DeviceSlice:
    """
    A slice of the model assigned to a device.
    
    Each device computes a subset of layers, then passes compressed
    activations to the next device.
    """
    device_id: str
    layer_start: int
    layer_end: int  # Exclusive
    
    # State
    cached_activations: Dict[int, np.ndarray] = field(default_factory=dict)
    pending_gradients: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    
    def get_layer_range(self) -> range:
        return range(self.layer_start, self.layer_end)
    
    def cache_activation(self, layer_idx: int, activation: np.ndarray):
        """Cache activation for backward pass"""
        self.cached_activations[layer_idx] = activation.copy()
    
    def clear_cache(self):
        """Clear cached activations to free memory"""
        self.cached_activations.clear()


class RingAllReduceMesh:
    """
    Implements Ring-AllReduce for distributed gradient aggregation.
    
    Each device only needs to communicate with its neighbors in the ring,
    making this efficient even on slow networks.
    """
    
    def __init__(self, devices: List[DeviceInfo]):
        self.devices = devices
        self.ring_order = self._create_ring()
        
    def _create_ring(self) -> List[str]:
        """Create ring topology from devices"""
        # Sort by compute score for balanced load
        sorted_devices = sorted(
            self.devices,
            key=lambda d: d.compute_score,
            reverse=True
        )
        return [d.device_id for d in sorted_devices]
    
    def allreduce(self, local_gradients: Dict[str, np.ndarray],
                  device_id: str) -> Dict[str, np.ndarray]:
        """
        Perform ring all-reduce for gradients.
        
        Each device sends a chunk to its right neighbor and receives
        from its left neighbor, accumulating gradients.
        """
        if len(self.ring_order) == 1:
            return local_gradients
        
        # Get position in ring
        my_idx = self.ring_order.index(device_id)
        left_idx = (my_idx - 1) % len(self.ring_order)
        right_idx = (my_idx + 1) % len(self.ring_order)
        
        left_neighbor = self.ring_order[left_idx]
        right_neighbor = self.ring_order[right_idx]
        
        # Divide gradients into chunks
        grad_items = list(local_gradients.items())
        n_chunks = len(self.ring_order)
        chunk_size = max(1, len(grad_items) // n_chunks)
        
        # Phase 1: Scatter-reduce
        # Each device accumulates gradients from all other devices
        accumulated = {}
        for i, (name, grad) in enumerate(grad_items):
            chunk_idx = i % n_chunks
            if chunk_idx == my_idx:
                accumulated[name] = grad.copy()
        
        # Simulate ring communication (in real impl, use network)
        for step in range(n_chunks - 1):
            send_chunk_idx = (my_idx - step) % n_chunks
            # Send chunk to right, receive from left
            # Accumulate received gradients
            pass  # Network communication happens here
        
        # Phase 2: All-gather
        # Distribute accumulated gradients to all devices
        all_gradients = {}
        for step in range(n_chunks - 1):
            # Send accumulated chunk to right
            # Receive chunk from left
            pass
        
        return accumulated
    
    def get_neighbors(self, device_id: str) -> Tuple[str, str]:
        """Get left and right neighbors in ring"""
        idx = self.ring_order.index(device_id)
        left = self.ring_order[(idx - 1) % len(self.ring_order)]
        right = self.ring_order[(idx + 1) % len(self.ring_order)]
        return left, right


class SwarmCoordinator:
    """
    Coordinates a swarm of devices for distributed training.
    
    Manages:
    - Device discovery and health monitoring
    - Layer assignment to devices
    - Data flow between devices
    - Fault tolerance and recovery
    """
    
    def __init__(self, 
                 n_layers: int,
                 enable_redundancy: bool = True,
                 checkpoint_interval: int = 100):
        self.n_layers = n_layers
        self.enable_redundancy = enable_redundancy
        self.checkpoint_interval = checkpoint_interval
        
        # Device management
        self.devices: Dict[str, DeviceInfo] = {}
        self.device_slices: Dict[str, DeviceSlice] = {}
        
        # Ring topology
        self.ring_mesh: Optional[RingAllReduceMesh] = None
        
        # Communication
        self.message_queues: Dict[str, queue.Queue] = {}
        self.compression_enabled = True
        
        # Training state
        self.current_step = 0
        self.global_gradients: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'data_transferred_mb': 0.0,
            'devices_joined': 0,
            'devices_failed': 0,
            'reassignments': 0,
        }
        
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Background threads
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._assignment_thread: Optional[threading.Thread] = None
        
    def register_device(self, device_info: DeviceInfo) -> bool:
        """Register a new device to the swarm"""
        with self._lock:
            self.devices[device_info.device_id] = device_info
            self.message_queues[device_info.device_id] = queue.Queue()
            self.stats['devices_joined'] += 1
            
            print(f"🖥️  Device {device_info.device_id} registered")
            print(f"   VRAM: {device_info.vram_gb}GB, RAM: {device_info.ram_gb}GB")
            print(f"   Compute: {device_info.compute_score:.1f}x")
            
            # Recompute assignments
            self._reassign_layers()
            
            return True
    
    def _reassign_layers(self):
        """Reassign layers to devices based on capabilities"""
        if not self.devices:
            return
        
        # Calculate total compute capacity
        total_compute = sum(d.compute_score for d in self.devices.values())
        
        # Distribute layers proportionally
        current_layer = 0
        device_list = list(self.devices.values())
        
        for device in device_list:
            # Calculate how many layers this device should handle
            layer_share = (device.compute_score / total_compute) * self.n_layers
            n_layers = max(1, int(round(layer_share)))
            
            # Ensure we don't exceed total
            if current_layer + n_layers > self.n_layers:
                n_layers = self.n_layers - current_layer
            
            # Assign slice
            slice_assignment = DeviceSlice(
                device_id=device.device_id,
                layer_start=current_layer,
                layer_end=current_layer + n_layers
            )
            
            self.device_slices[device.device_id] = slice_assignment
            device.assigned_layers = list(slice_assignment.get_layer_range())
            
            print(f"   {device.device_id}: layers {current_layer}-{current_layer + n_layers - 1}")
            
            current_layer += n_layers
            if current_layer >= self.n_layers:
                break
        
        # Update ring mesh
        self.ring_mesh = RingAllReduceMesh(device_list)
        self.stats['reassignments'] += 1
    
    def distribute_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute forward pass across the swarm.
        
        Each device computes its assigned layers and passes
        compressed activations to the next device.
        """
        current_activation = input_data
        
        # Sort devices by layer order
        sorted_slices = sorted(
            self.device_slices.values(),
            key=lambda s: s.layer_start
        )
        
        for device_slice in sorted_slices:
            device = self.devices.get(device_slice.device_id)
            if not device or device.status == DeviceStatus.OFFLINE:
                # Skip failed device - will use cached or recomputed
                print(f"⚠️  Device {device_slice.device_id} offline, skipping")
                continue
            
            # In real implementation, this would be RPC to remote device
            # For now, simulate local computation
            print(f"➡️  Forward through {device_slice.device_id}: "
                  f"layers {device_slice.layer_start}-{device_slice.layer_end - 1}")
            
            # Simulate processing
            # current_activation = self._compute_on_device(
            #     device, device_slice, current_activation
            # )
            
            # Cache for backward pass
            device_slice.cache_activation(device_slice.layer_end - 1, current_activation)
            
            # Compress for network transfer
            if self.compression_enabled:
                # compressed = tcl_compress(current_activation)
                # self.stats['data_transferred_mb'] += compressed_size
                pass
        
        return current_activation
    
    def distribute_backward(self, output_gradient: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute backward pass across the swarm.
        
        Gradients flow backwards through devices, each computing
        gradients for their assigned layers.
        """
        current_grad = output_gradient
        all_gradients = {}
        
        # Process in reverse order
        sorted_slices = sorted(
            self.device_slices.values(),
            key=lambda s: s.layer_start,
            reverse=True
        )
        
        for device_slice in sorted_slices:
            device = self.devices.get(device_slice.device_id)
            if not device or device.status == DeviceStatus.OFFLINE:
                continue
            
            print(f"⬅️  Backward through {device_slice.device_id}")
            
            # Compute gradients for this slice
            slice_gradients = self._compute_gradients_on_device(
                device, device_slice, current_grad
            )
            
            all_gradients.update(slice_gradients)
            
            # Update current grad for previous device
            # current_grad = propagate_gradient(current_grad)
        
        # All-reduce gradients across devices
        if self.ring_mesh and len(self.devices) > 1:
            for device_id in self.devices:
                all_gradients = self.ring_mesh.allreduce(all_gradients, device_id)
        
        return all_gradients
    
    def _compute_gradients_on_device(self, device: DeviceInfo,
                                     device_slice: DeviceSlice,
                                     output_grad: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients on a specific device"""
        # In real implementation, this would RPC to the device
        # For now, return empty gradients
        return {}
    
    def aggregate_gradients(self, local_gradients: Dict[str, np.ndarray],
                           device_id: str) -> Dict[str, np.ndarray]:
        """Aggregate gradients from all devices"""
        if not self.ring_mesh:
            return local_gradients
        
        return self.ring_mesh.allreduce(local_gradients, device_id)
    
    def heartbeat(self, device_id: str) -> bool:
        """Receive heartbeat from device"""
        with self._lock:
            if device_id in self.devices:
                self.devices[device_id].last_heartbeat = time.time()
                return True
            return False
    
    def check_device_health(self) -> List[str]:
        """Check health of all devices, return list of failed devices"""
        failed = []
        timeout = 30.0  # seconds
        
        with self._lock:
            for device_id, device in self.devices.items():
                time_since_heartbeat = time.time() - device.last_heartbeat
                
                if time_since_heartbeat > timeout:
                    if device.status != DeviceStatus.OFFLINE:
                        print(f"💔 Device {device_id} failed (no heartbeat)")
                        device.status = DeviceStatus.OFFLINE
                        self.stats['devices_failed'] += 1
                        failed.append(device_id)
                elif time_since_heartbeat > timeout / 2:
                    device.status = DeviceStatus.DEGRADED
                else:
                    device.status = DeviceStatus.ONLINE
        
        # Reassign if devices failed
        if failed and self.enable_redundancy:
            self._reassign_layers()
        
        return failed
    
    def get_swarm_stats(self) -> Dict[str, Any]:
        """Get statistics about the swarm"""
        return {
            **self.stats,
            'n_devices': len(self.devices),
            'n_online': sum(1 for d in self.devices.values() 
                          if d.status == DeviceStatus.ONLINE),
            'total_compute_capacity': sum(d.compute_score 
                                         for d in self.devices.values()),
            'total_vram_gb': sum(d.vram_gb for d in self.devices.values()),
            'layer_assignments': {
                d_id: {
                    'layers': list(s.get_layer_range()),
                    'n_params': 0  # Would calculate actual params
                }
                for d_id, s in self.device_slices.items()
            }
        }
    
    def shutdown(self):
        """Shutdown the swarm coordinator"""
        self._stop_event.set()
        print("🛑 Swarm coordinator shutting down")


class DistributedTrainingStep:
    """
    Encapsulates a single distributed training step across the swarm.
    """
    
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
        
    def run(self, batch_data: np.ndarray, target: np.ndarray,
            loss_fn: Callable) -> Dict[str, Any]:
        """
        Execute one distributed training step.
        
        1. Distribute forward pass
        2. Compute loss
        3. Distribute backward pass
        4. Aggregate gradients
        5. Update weights
        """
        start_time = time.time()
        
        # Forward pass
        output = self.coordinator.distribute_forward(batch_data)
        
        # Compute loss locally
        loss = loss_fn(output, target)
        
        # Backward pass
        output_grad = self._compute_output_gradient(output, target, loss_fn)
        gradients = self.coordinator.distribute_backward(output_grad)
        
        elapsed = time.time() - start_time
        
        return {
            'loss': float(loss),
            'gradients': gradients,
            'step_time': elapsed,
            'swarm_stats': self.coordinator.get_swarm_stats(),
        }
    
    def _compute_output_gradient(self, output: np.ndarray, target: np.ndarray,
                                 loss_fn: Callable) -> np.ndarray:
        """Compute gradient of loss w.r.t. output"""
        # Numerical gradient for flexibility
        eps = 1e-5
        grad = np.zeros_like(output)
        
        for i in range(min(100, output.size)):  # Sample for efficiency
            idx = np.unravel_index(i, output.shape)
            
            output_plus = output.copy()
            output_plus[idx] += eps
            loss_plus = loss_fn(output_plus, target)
            
            output_minus = output.copy()
            output_minus[idx] -= eps
            loss_minus = loss_fn(output_minus, target)
            
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad


class SwarmNetworkTransport:
    """
    Handles network communication between swarm devices.
    
    Uses compression and efficient protocols to work over
    consumer-grade networks (WiFi, Ethernet).
    """
    
    def __init__(self, listen_port: int = 9999):
        self.listen_port = listen_port
        self.peers: Dict[str, Tuple[str, int]] = {}  # device_id -> (ip, port)
        self.socket = None
        
    def start_server(self):
        """Start listening for connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', self.listen_port))
        self.socket.listen(10)
        print(f"📡 Swarm transport listening on port {self.listen_port}")
        
    def connect_to_peer(self, device_id: str, ip: str, port: int):
        """Connect to a peer device"""
        self.peers[device_id] = (ip, port)
        print(f"🔗 Connected to peer {device_id} at {ip}:{port}")
    
    def send_tensor(self, device_id: str, tensor: np.ndarray,
                   compression: bool = True) -> bool:
        """Send a tensor to a peer device"""
        if device_id not in self.peers:
            return False
        
        ip, port = self.peers[device_id]
        
        try:
            # Serialize
            data = pickle.dumps(tensor)
            
            # Compress if enabled
            if compression:
                import zlib
                data = zlib.compress(data)
            
            # Send
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((ip, port))
                sock.sendall(len(data).to_bytes(8, 'big'))
                sock.sendall(data)
            
            return True
        except Exception as e:
            print(f"❌ Failed to send to {device_id}: {e}")
            return False
    
    def receive_tensor(self) -> Optional[Tuple[str, np.ndarray]]:
        """Receive tensor from any peer"""
        if not self.socket:
            return None
        
        try:
            self.socket.settimeout(1.0)
            conn, addr = self.socket.accept()
            
            # Receive size
            size_data = conn.recv(8)
            size = int.from_bytes(size_data, 'big')
            
            # Receive data
            data = b''
            while len(data) < size:
                chunk = conn.recv(min(8192, size - len(data)))
                if not chunk:
                    break
                data += chunk
            
            # Decompress
            import zlib
            data = zlib.decompress(data)
            
            # Deserialize
            tensor = pickle.loads(data)
            
            return (addr[0], tensor)
        except socket.timeout:
            return None
        except Exception as e:
            print(f"❌ Receive error: {e}")
            return None
