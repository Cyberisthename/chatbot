# Synthetic GPU Miner - Quick Reference

## Installation

No special dependencies required beyond Python standard library. Optional:
```bash
pip install numpy  # For faster CPU mining
pip install pyopencl  # For OpenCL GPU support
pip install pycuda  # For CUDA GPU support
```

## Running the Miner

### Basic Usage
```bash
# Default: 3 jobs, 20-bit difficulty, 15s per job
python -m synthetic_gpu_miner.main

# Custom configuration
python -m synthetic_gpu_miner.main --jobs 5 --difficulty 22 --duration 30

# Help
python -m synthetic_gpu_miner.main --help
```

### Interactive Demo
```bash
# Full walkthrough of all features
python synthetic_gpu_miner/demo.py
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--jobs` | Number of mining jobs to run | 3 |
| `--difficulty` | Difficulty in leading zero bits (8-32) | 20 |
| `--duration` | Duration per job in seconds | 15.0 |
| `--verbose` | Enable verbose logging | False |

## Understanding the Output

### Device Detection
```
📊 Detected Devices (2):
   • cpu_0: CPU, perf_score=8.00, batch_size=4096
   • gpu_0: GPU, perf_score=600.00, batch_size=32768
```

- `perf_score`: Estimated performance relative to baseline CPU
- `batch_size`: Number of nonces processed per batch (auto-tuned)

### Mining Session
```
📦 Submitting job_1 (target: 0000ffffffffffff...)
   ✅ Share found! device=gpu_0, nonce=0012a4f3, diff=1.42
   📈 Status @ 5.0s: 45.23 MH/s, 1 shares, 226,150,400 hashes
```

- `nonce`: Solution that meets difficulty target
- `diff`: How much better than target (higher = rarer)
- `MH/s`: Millions of hashes per second

### Final Statistics
```
📊 Final Statistics:
   Total Hashes:   2,145,600,000
   Total Shares:   7
   Total Errors:   0
   Global Hashrate: 47.65 MH/s
   Uptime:         45.02s

   Device Performance:
      • cpu_0: 2.31 MH/s, latency=45.2ms, errors=0.000
      • gpu_0: 45.34 MH/s, latency=11.8ms, errors=0.000
```

## Difficulty Settings

| Bits | Target | Expected Shares per Million Hashes | Use Case |
|------|--------|--------------------------------------|----------|
| 8 | 0x00FF... | ~65,536 | Testing, demos |
| 16 | 0x0000FF... | ~256 | Quick validation |
| 20 | 0x00000F... | ~16 | Default (balanced) |
| 24 | 0x000000FF... | ~1 | Longer runs |
| 28 | 0x0000000F... | ~0.06 | Stress testing |
| 32 | 0x00000000FF... | ~0.004 | Production-like |

Higher difficulty = fewer shares found = longer runtime needed to see results.

## Architecture Layers

### 1. Protocol Layer (`protocol_layer.py`)
- Handles mining jobs (simulated Stratum protocol)
- Receives block templates
- Submits valid shares

### 2. Hash Core Layer (`hash_core.py`)
- Executes SHA-256 double-hash mining
- CPU: Multi-threaded with optional NumPy
- GPU: Simulated (4x speedup over CPU)

### 3. Precompute Cache Layer (`precompute_cache.py`)
- Computes midstates from fixed header parts
- Caches constants (K table, IV)
- Eliminates ~60-70% of redundant work

### 4. Synthetic GPU Layer (`scheduler.py`)
- Main scheduler orchestrating all work
- Maintains logical infinite work queue
- Adaptive batch sizing
- Heterogeneous device management

### 5. Telemetry Layer (`telemetry.py`)
- Tracks performance metrics
- Auto-tunes batch sizes
- Provides statistics

## Key Concepts

### WorkUnit
Smallest unit of mining work:
- `job_id`: Which job it belongs to
- `midstate_id`: Reference to precomputed state
- `nonce_start`: Starting nonce value
- `nonce_count`: How many nonces to check

### Batch
Collection of WorkUnits for a single device:
- All units must share same `job_id` and `midstate_id`
- Size adapts based on device performance
- Processed atomically

### Midstate
Precomputed SHA-256 state from fixed header parts:
- Computed once per job
- Reused for millions of nonces
- Dramatically reduces computation

### Adaptive Tuning
System continuously measures and adjusts:
- If batch too slow → reduce size
- If batch too fast → increase size
- If device idle → give more work
- If device errors → reduce load

## Performance Tips

### For Maximum Speed
1. Install NumPy: `pip install numpy`
2. Use lower difficulty for testing (8-12 bits)
3. Adjust batch sizes manually if needed (modify device_manager.py)
4. Run on multi-core CPU for best results

### For Longer Runs
1. Use higher difficulty (24-32 bits)
2. Increase duration (--duration 60 or more)
3. Increase number of jobs (--jobs 10)

### For Testing Features
1. Use very low difficulty (8 bits)
2. Short duration (--duration 5)
3. Run demo.py for interactive walkthrough

## Extending the System

### Add Real GPU Support
1. Implement CUDA kernel (see `gpu_kernels.cu`)
2. Create GPU executor in `hash_core.py`
3. Use PyCUDA or PyOpenCL to launch kernels

### Add Real Pool Connection
1. Extend `protocol_layer.py` with Stratum client
2. Connect to pool: `stratum+tcp://pool:port`
3. Authenticate worker
4. Handle job updates and share submissions

### Add New Hash Algorithm
1. Create new hash core class
2. Implement algorithm-specific mining logic
3. Update precompute cache for algorithm optimizations

## Troubleshooting

### No devices detected
- System will auto-detect CPU
- If GPU not detected, check drivers (CUDA/OpenCL)
- GPU detection requires nvidia-smi or pyopencl

### Low hashrate
- Expected on Python implementation (~1-10 MH/s CPU)
- For production, need native CUDA/OpenCL kernels
- Install NumPy for ~2x speedup

### No shares found
- Difficulty too high for duration
- Lower difficulty or increase duration
- Expected shares = (total_hashes) / (2^(256-difficulty_bits))

### Errors in output
- Check Python version (3.7+)
- Check for missing dependencies
- Try verbose mode: `--verbose`

## File Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization |
| `main.py` | Entry point, CLI interface |
| `demo.py` | Interactive demonstration |
| `work_unit.py` | Core data structures |
| `device_manager.py` | Device detection and management |
| `hash_core.py` | SHA-256 mining implementation |
| `precompute_cache.py` | Midstate caching |
| `protocol_layer.py` | Mining protocol abstraction |
| `scheduler.py` | Main scheduler logic |
| `telemetry.py` | Performance monitoring |
| `gpu_kernels.cu` | CUDA kernel reference |
| `README.md` | User guide |
| `ARCHITECTURE.md` | Deep technical documentation |
| `QUICKREF.md` | This quick reference |

## Further Reading

- **README.md**: Complete user guide with examples
- **ARCHITECTURE.md**: In-depth design and implementation details
- **gpu_kernels.cu**: CUDA kernel implementation reference

## Support

This is a research/educational project. For questions:
1. Check documentation in `synthetic_gpu_miner/`
2. Review code comments
3. Run demo.py for interactive examples

Remember: This demonstrates architectural concepts, not production mining!
