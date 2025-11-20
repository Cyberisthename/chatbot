import platform
import subprocess
from typing import Dict, List, Optional

from .work_unit import Device, DeviceType


class DeviceManager:
    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self._detect_devices()

    def _detect_devices(self) -> None:
        self._detect_cpu()
        self._detect_gpu()

    def _detect_cpu(self) -> None:
        try:
            cpu_count = self._get_cpu_count()
            cpu_id = "cpu_0"
            perf_score = self._estimate_cpu_performance(cpu_count)
            self.devices[cpu_id] = Device(
                id=cpu_id,
                type=DeviceType.CPU,
                perf_score=perf_score,
                batch_size=4096,
                max_batch_size=65536,
                min_batch_size=512,
                metadata={"cpu_count": cpu_count, "platform": platform.processor()},
            )
            print(f"✓ Detected CPU: {cpu_count} cores, perf_score={perf_score:.2f}")
        except Exception as e:
            print(f"✗ CPU detection failed: {e}")

    def _get_cpu_count(self) -> int:
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except Exception:
            return 1

    def _estimate_cpu_performance(self, cpu_count: int) -> float:
        base_perf = 1.0
        has_avx2 = self._check_avx2()
        has_avx512 = self._check_avx512()
        if has_avx512:
            base_perf *= 4.0
        elif has_avx2:
            base_perf *= 2.0
        return base_perf * cpu_count

    def _check_avx2(self) -> bool:
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx2' in info.get('flags', [])
        except ImportError:
            if platform.system() == "Linux":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        return 'avx2' in f.read()
                except Exception:
                    pass
            return False

    def _check_avx512(self) -> bool:
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            return any('avx512' in flag for flag in flags)
        except ImportError:
            if platform.system() == "Linux":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        return 'avx512' in f.read()
                except Exception:
                    pass
            return False

    def _detect_gpu(self) -> None:
        gpus = self._detect_cuda_gpus()
        if not gpus:
            gpus = self._detect_opencl_gpus()
        for idx, gpu_info in enumerate(gpus):
            gpu_id = f"gpu_{idx}"
            self.devices[gpu_id] = Device(
                id=gpu_id,
                type=DeviceType.GPU,
                perf_score=gpu_info['perf_score'],
                batch_size=32768,
                max_batch_size=524288,
                min_batch_size=4096,
                metadata=gpu_info,
            )
            print(f"✓ Detected GPU {idx}: {gpu_info.get('name', 'Unknown')}, perf_score={gpu_info['perf_score']:.2f}")

    def _detect_cuda_gpus(self) -> List[Dict]:
        gpus = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        idx, name, memory_mb = parts[0], parts[1], parts[2]
                        memory_gb = float(memory_mb) / 1024.0
                        perf_score = self._estimate_gpu_performance(name, memory_gb)
                        gpus.append({
                            'index': int(idx),
                            'name': name,
                            'memory_gb': memory_gb,
                            'perf_score': perf_score,
                            'backend': 'CUDA',
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return gpus

    def _detect_opencl_gpus(self) -> List[Dict]:
        gpus = []
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                for idx, device in enumerate(platform.get_devices()):
                    if device.type == cl.device_type.GPU:
                        name = device.name
                        memory_gb = device.global_mem_size / (1024**3)
                        perf_score = self._estimate_gpu_performance(name, memory_gb)
                        gpus.append({
                            'index': idx,
                            'name': name,
                            'memory_gb': memory_gb,
                            'perf_score': perf_score,
                            'backend': 'OpenCL',
                        })
        except ImportError:
            pass
        except Exception:
            pass
        return gpus

    def _estimate_gpu_performance(self, name: str, memory_gb: float) -> float:
        name_lower = name.lower()
        base_score = 100.0
        if 'rtx 4090' in name_lower:
            base_score = 1000.0
        elif 'rtx 4080' in name_lower:
            base_score = 800.0
        elif 'rtx 4070' in name_lower:
            base_score = 600.0
        elif 'rtx 3090' in name_lower:
            base_score = 700.0
        elif 'rtx 3080' in name_lower:
            base_score = 600.0
        elif 'rtx 3070' in name_lower:
            base_score = 500.0
        elif 'rtx' in name_lower:
            base_score = 400.0
        elif 'gtx' in name_lower:
            base_score = 200.0
        elif 'a100' in name_lower:
            base_score = 1200.0
        elif 'v100' in name_lower:
            base_score = 800.0
        elif 't4' in name_lower:
            base_score = 300.0
        else:
            base_score = max(100.0, memory_gb * 50)
        return base_score

    def get_all_devices(self) -> List[Device]:
        return list(self.devices.values())

    def get_device(self, device_id: str) -> Optional[Device]:
        return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        return [d for d in self.devices.values() if d.type == device_type]

    def get_idle_devices(self) -> List[Device]:
        return [d for d in self.devices.values() if not d.is_busy]

    def update_device_profile(self, device_id: str, perf_score: Optional[float] = None,
                              last_latency: Optional[float] = None, error_rate: Optional[float] = None) -> None:
        device = self.devices.get(device_id)
        if not device:
            return
        if perf_score is not None:
            device.perf_score = perf_score
        if last_latency is not None:
            device.last_latency = last_latency
        if error_rate is not None:
            device.error_rate = error_rate
