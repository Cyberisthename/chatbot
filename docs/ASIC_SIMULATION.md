# Virtual ASIC Device Simulation

## Overview

The Jarvis-5090X system now includes support for **virtual ASIC mining devices** with realistic performance modeling. This enables research, analysis, and experimentation with ultra-high-throughput mining scenarios without requiring actual ASIC hardware.

## Features

### 1. Virtual ASIC Device Representation

ASIC devices are represented using the `AdapterDevice` class with special metadata:

```python
from jarvis5090x import AdapterDevice, DeviceKind, OperationKind

asic_device = AdapterDevice(
    id="asic_0",
    label="Virtual ASIC Miner",
    kind=DeviceKind.VIRTUAL,
    perf_score=10_000.0,      # High score ensures scheduler priority
    max_concurrency=128,
    capabilities={OperationKind.HASHING},
    metadata={
        "asic_model": "VA-1",
        "hashes_per_second": 1e12,   # 1 TH/s throughput
        "latency_overhead_ms": 0.1,
    },
)
```

### 2. Performance Simulation

The ASIC simulation uses a realistic performance model:

- **Throughput**: Specified via `hashes_per_second` in device metadata
- **Latency**: Computed as `nonce_count / F + overhead` where F is the hashrate
- **Correctness**: Always computes at least one real hash for verification

When a hashing job is submitted to an ASIC device:

```python
result = jarvis.submit("hashing", "job_signature", {
    "header_prefix": b"\x01" * 76,
    "nonce_start": 0,
    "nonce_count": 1_000_000,
    "target": (1 << 224) - 1,
})
```

The result includes simulation details:

```python
{
    "hash": "ad5deab44f7a19144e3d75f588544f8c...",
    "hashes_processed": 1_000_000,
    "device_id": "asic_0",
    "simulated": True,
    "asic_model": "VA-1",
    "simulated_latency_ms": 1.1,
    "effective_hashrate_hs": 1e12,
    "algorithm": "sha256",
}
```

### 3. Brute-Force Cryptographic Analysis

The system includes tools for analyzing brute-force feasibility:

```python
from jarvis5090x import (
    is_bruteforce_feasible,
    bruteforce_threshold_bits,
    estimate_time_to_crack,
)

# Check if 64-bit key is crackable in 1 year
hashrate = 1e12  # 1 TH/s
feasible = is_bruteforce_feasible(64, hashrate, 31536000)

# Find maximum bits crackable in 1 day
max_bits = bruteforce_threshold_bits(hashrate, 86400)

# Estimate time to crack 80-bit key
time_seconds = estimate_time_to_crack(80, hashrate)
```

**Mathematical Model:**

- Search space: `2^n` where n is key size in bits
- Computational budget: `B = F × τ` where F is hashrate and τ is time
- Feasibility: `2^n ≤ F × τ`
- Maximum bits: `n_max = log₂(F × τ)`

### 4. Calibration & Real-World Mapping

Convert simulated performance to real-world equivalents:

```python
from jarvis5090x import (
    compute_scale_factor,
    to_real_hashrate,
    combine_hashrates_real,
)

# Calibrate: measure real GPU and compare to simulation
H_real_measured = 80e6    # 80 MH/s from actual hardware
H_real_sim = 1e12         # 1 TH/s from simulation

scale_factor = compute_scale_factor(H_real_measured, H_real_sim)
# α ≈ 8e-5

# Convert virtual to real
H_virtual_real = to_real_hashrate(1e12, scale_factor)
# ≈ 80 MH/s equivalent

# Combine real GPU + virtual devices
H_total = combine_hashrates_real(
    H_real_gpu_hs=80e6,
    H_virtual_sim_hs=1e12,
    scale_factor=scale_factor,
)
# ≈ 160 MH/s total
```

### 5. Mining Reward Estimation

Estimate mining earnings based on hashrate:

```python
from jarvis5090x import estimate_daily_reward

daily_btc = estimate_daily_reward(
    your_hashrate_hs=1e12,        # 1 TH/s
    network_hashrate_hs=400e18,   # 400 EH/s (Bitcoin network)
    blocks_per_day=144,           # Bitcoin: ~10 min blocks
    block_reward_coins=6.25,      # Current Bitcoin reward
)
# ≈ 0.00000225 BTC/day
```

**Formula:**

```
daily_reward = (your_hashrate / network_hashrate) × blocks_per_day × block_reward
```

### 6. Integrated Performance Analysis

Run comprehensive analysis with a single function:

```python
from jarvis5090x import analyze_mining_performance

analysis = analyze_mining_performance(
    jarvis,
    workload_nonces=1_000_000,
)

# Returns:
{
    "workload_nonces": 1_000_000,
    "elapsed_s": 0.0015,
    "hashrate_hs": 666666666.67,
    "hashrate_formatted": "666.67 MH/s",
    "device_id": "asic_0",
    "simulated": True,
    "asic_model": "VA-1",
    "bruteforce_analysis": {
        "1_second": 29.3,
        "1_minute": 35.2,
        "1_hour": 41.1,
        "1_day": 45.8,
        "1_year": 54.3,
    },
}
```

## Usage Examples

### Example 1: Create an ASIC Mining Cluster

```python
from jarvis5090x import (
    AdapterCluster,
    AdapterDevice,
    DeviceKind,
    Jarvis5090X,
    OperationKind,
)

# Define devices
devices = [
    AdapterDevice(
        id="cpu_0",
        label="CPU Miner",
        kind=DeviceKind.CPU,
        perf_score=50.0,
        max_concurrency=8,
        capabilities={OperationKind.HASHING},
        metadata={"hashes_per_second": 1e6},  # 1 MH/s
    ),
    AdapterDevice(
        id="gpu_0",
        label="GPU Miner",
        kind=DeviceKind.GPU,
        perf_score=500.0,
        max_concurrency=32,
        capabilities={OperationKind.HASHING},
        metadata={"hashes_per_second": 1e9},  # 1 GH/s
    ),
    AdapterDevice(
        id="asic_0",
        label="ASIC Miner",
        kind=DeviceKind.VIRTUAL,
        perf_score=10_000.0,
        max_concurrency=128,
        capabilities={OperationKind.HASHING},
        metadata={
            "asic_model": "VA-1",
            "hashes_per_second": 1e12,  # 1 TH/s
            "latency_overhead_ms": 0.1,
        },
    ),
]

# Initialize Jarvis
cluster = AdapterCluster(devices)
jarvis = Jarvis5090X(devices, adapter_cluster=cluster)

# Submit mining job (scheduler will pick ASIC due to high perf_score)
result = jarvis.submit("hashing", "mining_job", {
    "header_prefix": b"\x01" * 76,
    "nonce_start": 0,
    "nonce_count": 1_000_000,
    "target": (1 << 224) - 1,
})

print(f"Device: {result['device_id']}")
print(f"Hashrate: {result['effective_hashrate_hs']:.2e} H/s")
```

### Example 2: Security Analysis

```python
from jarvis5090x import (
    format_hashrate,
    format_time,
    estimate_time_to_crack,
)

# Analyze different ASIC configurations
configs = [
    ("Single ASIC", 1e12),
    ("10 ASICs", 1e13),
    ("100 ASICs", 1e14),
    ("Mining Farm", 1e15),
]

print("Time to crack 64-bit key:")
for label, hashrate in configs:
    time_s = estimate_time_to_crack(64, hashrate)
    print(f"  {label:15s} ({format_hashrate(hashrate):>12s}): {format_time(time_s)}")

# Output:
# Single ASIC     (     1.00 TH/s): 106.75 days
# 10 ASICs        (    10.00 TH/s): 10.68 days
# 100 ASICs       (   100.00 TH/s): 1.07 days
# Mining Farm     (     1.00 PH/s): 2.56 hours
```

### Example 3: Profitability Estimation

```python
from jarvis5090x import estimate_daily_reward, format_hashrate

# Bitcoin network parameters (example)
network_hashrate = 400e18  # 400 EH/s
btc_price_usd = 40000
power_cost_kwh = 0.10

# Your ASIC configuration
asic_hashrate = 1e12      # 1 TH/s
power_watts = 1500

# Calculate rewards
daily_btc = estimate_daily_reward(
    asic_hashrate,
    network_hashrate,
    blocks_per_day=144,
    block_reward_coins=6.25,
)

# Calculate profit
daily_revenue = daily_btc * btc_price_usd
daily_power_cost = (power_watts / 1000) * 24 * power_cost_kwh
daily_profit = daily_revenue - daily_power_cost

print(f"ASIC: {format_hashrate(asic_hashrate)}")
print(f"Daily BTC: {daily_btc:.8f}")
print(f"Revenue: ${daily_revenue:.2f}")
print(f"Power Cost: ${daily_power_cost:.2f}")
print(f"Profit: ${daily_profit:.2f}")
```

## API Reference

### Device Creation

**`AdapterDevice` with ASIC metadata:**

- `id`: Unique device identifier
- `label`: Human-readable name
- `kind`: `DeviceKind.VIRTUAL` (recommended for ASICs)
- `perf_score`: High value (e.g., 10,000+) for scheduler priority
- `max_concurrency`: Number of parallel jobs
- `capabilities`: `{OperationKind.HASHING}`
- `metadata`:
  - `asic_model`: Model identifier (e.g., "VA-1")
  - `hashes_per_second`: Throughput in H/s (e.g., 1e12 for 1 TH/s)
  - `latency_overhead_ms`: Overhead in milliseconds

### Analysis Functions

**`is_bruteforce_feasible(n_bits, hashrate_hs, time_seconds) -> bool`**

Check if key size is crackable within time budget.

**`bruteforce_threshold_bits(hashrate_hs, time_seconds) -> float`**

Calculate maximum crackable key size in bits.

**`estimate_time_to_crack(key_bits, hashrate_hs, success_probability=0.5) -> float`**

Estimate time (seconds) to crack a key with given probability.

**`analyze_mining_performance(jarvis, workload_nonces, device_id=None) -> dict`**

Run comprehensive performance analysis.

### Calibration Functions

**`compute_scale_factor(H_real_measured_hs, H_real_sim_hs) -> float`**

Compute calibration factor for virtual→real conversion.

**`to_real_hashrate(sim_hashrate_hs, scale_factor) -> float`**

Convert simulated hashrate to real-world equivalent.

**`combine_hashrates_real(H_real_gpu_hs, H_virtual_sim_hs, scale_factor) -> float`**

Combine real and virtual hashrates.

### Reward Functions

**`estimate_daily_reward(your_hashrate_hs, network_hashrate_hs, blocks_per_day, block_reward_coins) -> float`**

Estimate daily mining reward in coins.

### Utility Functions

**`format_hashrate(hashrate_hs) -> str`**

Format hashrate with SI prefix (e.g., "1.50 TH/s").

**`format_time(seconds) -> str`**

Format duration (e.g., "2.5 days", "3.2 years").

## Demos

### Run ASIC Mining Demo

```bash
python demo_asic_mining.py
```

Features:
- ASIC device creation
- Performance comparison (CPU vs GPU vs ASIC)
- Brute-force cryptographic analysis
- Calibration and real-world mapping
- Mining reward estimation
- Integrated performance analysis

### Run Standard Demo with ASIC

```bash
python demo_5090x.py
```

The standard Jarvis-5090X demo now includes an ASIC device alongside CPU and GPU devices.

### Run Tests

```bash
python test_asic_simulation.py
```

## Use Cases

### 1. **Cryptographic Research**

Analyze the security of hash-based systems against ASIC mining attacks:

- Model real-world ASIC capabilities
- Estimate time/cost to break various key sizes
- Compare CPU/GPU/ASIC performance

### 2. **Mining Economics**

Simulate mining profitability without hardware investment:

- Test different hashrate configurations
- Model network difficulty changes
- Optimize hardware deployment strategies

### 3. **Hardware Co-Design**

Design and test virtual mining architectures:

- Experiment with hypothetical ASIC designs
- Benchmark against current hardware
- Validate performance models

### 4. **Education**

Teach concepts in:

- Cryptocurrency mining
- Computational complexity
- Hardware acceleration
- Performance modeling

## Performance Model Details

### Simulation Accuracy

The ASIC simulation provides:

1. **Correct hash computation**: Always computes real SHA-256 hashes
2. **Realistic latency**: Based on throughput and overhead
3. **Throughput modeling**: Linear scaling with nonce count
4. **Scheduler integration**: Devices compete for jobs based on perf_score

### Limitations

- **Not a real miner**: Does not connect to pools or blockchain networks
- **Simplified model**: Real ASICs have more complex performance characteristics
- **No power/thermal modeling**: Does not simulate power consumption or thermal throttling
- **Static throughput**: Real devices may vary with temperature, voltage, etc.

### Calibration Methodology

To map simulation to reality:

1. Run benchmark on real hardware (GPU/ASIC)
2. Run same benchmark in Jarvis simulation
3. Compute scale factor: `α = H_real / H_sim`
4. Apply to future simulations: `H_real_equiv = H_sim × α`

## Integration with Synthetic GPU Miner

The ASIC simulation integrates with the existing `synthetic_gpu_miner` package:

- Uses `HashCore` for non-ASIC devices
- Falls back to simulation for ASIC devices
- Shares same scheduler and cluster infrastructure
- Compatible with existing mining demos

## Future Enhancements

Potential improvements:

1. **Dynamic hashrate**: Simulate variable throughput based on network difficulty
2. **Power modeling**: Add power consumption and efficiency metrics
3. **Thermal simulation**: Model temperature effects on performance
4. **Pool integration**: Connect to real mining pools (testnet only)
5. **Multiple algorithms**: Support SHA-256d, Ethash, Equihash, etc.
6. **Network effects**: Simulate impact of network hashrate changes

## License

This ASIC simulation feature is part of the Jarvis-5090X virtual GPU system and follows the same license as the main project.

## Disclaimer

**This is a simulation for research and educational purposes only.**

- Virtual ASIC devices do not represent real hardware
- Performance numbers are theoretical
- Not suitable for production mining
- Mining cryptocurrency requires appropriate hardware and electricity costs

---

For more information, see:
- `demo_asic_mining.py` - Comprehensive demo
- `test_asic_simulation.py` - Test suite
- `jarvis5090x/asic_utils.py` - Implementation
