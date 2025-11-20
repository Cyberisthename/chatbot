# Synthetic GPU Miner - Advanced Mining Scheduler

A research-grade hybrid CPU+GPU mining architecture that demonstrates the "Infinite Capacity" concept applied to cryptocurrency mining. This system breaks mining work into micro-tasks, precomputes everything possible, and adaptively schedules across all available compute resources.

## 🏗️ Architecture Overview

### Five-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│  1. Protocol Layer (Stratum/Pool Communication)             │
├─────────────────────────────────────────────────────────────┤
│  2. Hash Core Layer (SHA-256 GPU/CPU Implementation)        │
├─────────────────────────────────────────────────────────────┤
│  3. Precomputation & Cache Layer (Midstates & Constants)    │
├─────────────────────────────────────────────────────────────┤
│  4. Synthetic GPU Layer (Intelligent Scheduler)             │
├─────────────────────────────────────────────────────────────┤
│  5. Control & Telemetry Layer (Performance Monitoring)      │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

**WorkUnit**: The smallest unit of mining work
- Contains: job_id, midstate_id, nonce_start, nonce_count
- Can be split and recombined dynamically

**Batch**: Collection of WorkUnits for a single device
- All units share the same job_id and midstate_id
- Size adapts based on device performance

**Device**: Represents GPU or CPU compute resource
- Tracks performance score, batch size, latency, error rate
- Automatically tunes batch size for optimal throughput

**Midstate**: Precomputed hash state for fixed header parts
- Computed once per job, reused millions of times
- Dramatically reduces redundant computation

**Synthetic Parallelism**: Logical infinite task queue
- Physical devices pull work on-demand
- Adaptive load balancing based on real-time performance

## 📊 How It Works

### Mining Loop Flow

```
1. New Job Arrives
   ├─> Protocol Layer receives block template
   └─> Extract: header_prefix, target, job_id

2. Precompute Phase
   ├─> Compute midstate from header_prefix
   ├─> Upload to GPU constant memory
   └─> Build CPU-side lookup tables

3. Generate Logical Work
   ├─> Split nonce range (0 to 2^32-1)
   └─> Create millions of tiny WorkUnits

4. Schedule to Devices
   ├─> For each idle device:
   │   ├─> Pull N WorkUnits (dynamic batch size)
   │   ├─> Pack into Batch
   │   └─> Launch kernel/thread
   └─> Repeat continuously

5. Collect Results
   ├─> Monitor device completion
   ├─> Extract valid shares (hash < target)
   └─> Submit to pool

6. Adapt & Tune
   ├─> Measure: hashrate, latency, error rate
   ├─> Adjust batch sizes
   └─> Rebalance CPU/GPU work ratio
```

### Adaptive Load Balancing

The system continuously measures performance and adapts:

- **Too slow?** → Reduce batch size
- **Too fast?** → Increase batch size  
- **High latency?** → Reduce work allocation
- **Low error rate?** → Increase work allocation
- **GPU idle?** → Give more work
- **CPU underutilized?** → Shift work from GPU

## 🚀 Quick Start

### Installation

```bash
pip install numpy
# Optional: for GPU support
pip install pyopencl  # or pycuda
```

### Run Demo

```bash
# Basic demo (3 jobs, 20-bit difficulty, 15s each)
python -m synthetic_gpu_miner.main

# Custom configuration
python -m synthetic_gpu_miner.main --jobs 5 --difficulty 22 --duration 30

# Verbose mode
python -m synthetic_gpu_miner.main --verbose
```

### Output Example

```
================================================================================
🚀 SYNTHETIC GPU MINER - Starting...
================================================================================

⚙️  Configuration:
   • Number of jobs: 3
   • Difficulty: 20 leading zero bits
   • Duration: 15.0s per job

📊 Detected Devices (2):
   • cpu_0: CPU, perf_score=8.00, batch_size=4096
   • gpu_0: GPU, perf_score=600.00, batch_size=32768

================================================================================

📦 Submitting job_1 (target: 0000ffffffffffffffffffffffffffff...)
   ✅ Share found! device=gpu_0, nonce=0012a4f3, diff=1.42
   📈 Status @ 5.0s: 45.23 MH/s, 1 shares, 226,150,400 hashes
   ✅ Share found! device=gpu_0, nonce=003f7821, diff=1.18
   📈 Status @ 10.0s: 47.81 MH/s, 2 shares, 478,100,000 hashes
   ⏱️  Job job_1 completed (15.0s)

...

================================================================================
✅ Mining session completed in 45.02s

📊 Final Statistics:

   Total Hashes:   2,145,600,000
   Total Shares:   7
   Total Errors:   0
   Global Hashrate: 47.65 MH/s
   Uptime:         45.02s

   Device Performance:
      • cpu_0: 2.31 MH/s, latency=45.2ms, errors=0.000
      • gpu_0: 45.34 MH/s, latency=11.8ms, errors=0.000

================================================================================
```

## 📁 Module Reference

### `work_unit.py`
Core data structures: WorkUnit, Batch, Device, DeviceType

### `device_manager.py`
Device detection and capability assessment
- Auto-detects CPUs (with AVX2/AVX512 support)
- Auto-detects GPUs (CUDA via nvidia-smi, OpenCL via pyopencl)
- Estimates performance scores

### `hash_core.py`
Hash computation engine
- CPU: Multi-threaded SHA-256 with NumPy vectorization
- GPU: Simulated kernel execution (4x speedup over CPU)
- Extensible for real CUDA/OpenCL kernels

### `precompute_cache.py`
Midstate computation and caching
- Computes SHA-256 midstate from fixed header parts
- Caches by midstate_id to avoid recomputation
- Auto-clears old midstates on job change

### `protocol_layer.py`
Mining protocol abstraction
- Receives mining jobs (simulated or real Stratum)
- Submits shares to pool
- Job dispatch with callback registration

### `scheduler.py`
The heart of the synthetic GPU system
- Maintains logical infinite work queue
- Schedules batches to idle devices
- Collects results asynchronously
- Triggers adaptive tuning

### `telemetry.py`
Performance monitoring and auto-tuning
- Tracks per-device: hashrate, latency, error rate
- Maintains rolling window of performance snapshots
- Estimates optimal batch sizes
- Provides global and per-device statistics

### `main.py`
Entry point and demo application

## 🎯 Design Goals

### 1. **Maximize Utilization**
Keep all devices as close to 100% busy as possible through:
- Asynchronous task dispatch
- Prefetching next batches
- Overlapping compute and data transfer

### 2. **Minimize Redundant Work**
Precompute everything that can be precomputed:
- Hash constants (SHA-256 K table)
- Initial hash states
- Midstates from fixed header parts
- Message schedule patterns

### 3. **Adaptive Performance**
Continuously learn and optimize:
- Measure real device performance
- Adjust batch sizes dynamically
- Rebalance work distribution
- Detect and compensate for errors

### 4. **Device Agnostic**
Treat all compute as a unified pool:
- GPU and CPU are just different device types
- Same WorkUnit format for all
- Scheduler doesn't care about hardware details

### 5. **Scalable Architecture**
Design for future expansion:
- Easy to add new device types (FPGAs, ASICs)
- Easy to add new hash algorithms
- Easy to integrate with real mining pools

## 🔬 Research Applications

This architecture demonstrates several advanced concepts:

### Synthetic Parallelism
The "infinite capacity" illusion created by:
- Logical task space larger than physical memory
- On-demand task generation
- Dynamic task sizing and splitting

### Heterogeneous Computing
Unified orchestration of diverse compute resources:
- Different performance characteristics (CPU vs GPU)
- Different latency profiles
- Different error rates
- Automatic load balancing

### Adaptive Systems
Real-time performance optimization:
- Online learning from performance metrics
- Automated parameter tuning
- Self-correcting behavior

### Resource Management
Efficient utilization of limited resources:
- Work stealing between devices
- Priority-based scheduling
- Deadlock prevention

## 🛠️ Extending the System

### Adding Real GPU Kernels

Replace the simulated GPU execution in `hash_core.py` with real CUDA/OpenCL:

```python
def _execute_gpu_batch(self, batch, midstate_payload, target):
    # Load CUDA kernel
    kernel = self.cuda_module.get_function("sha256_mine")
    
    # Allocate device memory
    d_midstate = cuda.mem_alloc(midstate_payload['partial_state'])
    d_results = cuda.mem_alloc(8 * 1024)  # Result buffer
    
    # Launch kernel
    kernel(d_midstate, np.uint32(batch.work_units[0].nonce_start),
           np.uint32(batch.total_nonce_count), d_results,
           block=(256, 1, 1), grid=(batch.total_nonce_count // 256, 1))
    
    # Collect results
    ...
```

### Adding Real Pool Connection

Replace the simulated protocol in `protocol_layer.py` with real Stratum:

```python
import socket
import json

class StratumProtocol(ProtocolLayer):
    def __init__(self, pool_url, pool_port, worker):
        super().__init__()
        self.socket = socket.create_connection((pool_url, pool_port))
        self._authenticate(worker)
    
    def _receive_job(self):
        data = self.socket.recv(4096)
        message = json.loads(data)
        if message['method'] == 'mining.notify':
            job = self._parse_stratum_job(message['params'])
            self.submit_job(job)
```

### Adding New Hash Algorithms

Extend `hash_core.py` with new algorithms:

```python
class ScryptHashCore(HashCore):
    def _mine_range(self, header_prefix, nonce_start, nonce_count, target):
        # Implement Scrypt mining
        ...
```

## 📈 Performance Expectations

### Theoretical Limits

For SHA-256 mining:
- Modern CPU (16 cores, AVX2): ~2-10 MH/s
- Consumer GPU (RTX 3080): ~300-500 MH/s (simulated)
- Professional GPU (A100): ~1000+ MH/s (simulated)

### Actual Performance

This implementation focuses on **architecture** over raw speed:
- Demonstrates concepts, not production mining
- CPU-bound by Python overhead
- GPU simulation (no real kernel execution)

For production mining, you would:
1. Implement native CUDA/OpenCL kernels
2. Use C++/Rust for scheduler
3. Optimize memory transfers
4. Use ASIC-specific optimizations

## ⚠️ Important Notes

### Not for Production Mining

This is a **research and educational project**:
- Demonstrates architectural concepts
- Not competitive with real mining hardware
- GPU execution is simulated (CPU-based)

### Real Mining Requirements

To mine profitably, you need:
- **ASICs**: Custom silicon for specific algorithms
- **Pool access**: Real Stratum protocol implementation
- **Optimization**: Native code, not Python
- **Scale**: Thousands of devices, not one laptop

### What This Project IS

✅ Educational demonstration of advanced scheduling  
✅ Research platform for heterogeneous computing  
✅ Proof of concept for "synthetic GPU" architecture  
✅ Foundation for building real mining software  

### What This Project IS NOT

❌ Production-ready mining software  
❌ Profitable on consumer hardware  
❌ Competitive with commercial miners  
❌ Magic infinite compute

## 📚 Further Reading

- **Bitcoin Mining**: Understanding the protocol and economics
- **CUDA Programming**: Building real GPU kernels
- **Heterogeneous Computing**: Managing diverse compute resources
- **Task Scheduling**: Advanced parallel computing techniques
- **Performance Optimization**: Profiling and tuning strategies

## 🤝 Contributing

This project welcomes contributions:
- Real GPU kernel implementations
- Additional hash algorithm support
- Real Stratum protocol integration
- Performance optimizations
- Documentation improvements

## 📄 License

Part of the Jarvis AI project. See main repository LICENSE.

---

**Built with the vision of turning limited hardware into infinite capability through intelligent software architecture.**
