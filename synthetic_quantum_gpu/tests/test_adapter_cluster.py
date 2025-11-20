import time
import unittest

from synthetic_quantum_gpu.adapter_cluster import SyntheticAdapterCluster
from synthetic_quantum_gpu.types import Device, DeviceType, WorkUnit


class TestSyntheticAdapterCluster(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = [
            Device(id="cpu", type=DeviceType.CPU, batch_size=4, perf_score=5.0),
            Device(id="gpu", type=DeviceType.GPU, batch_size=4, perf_score=10.0),
        ]
        self.processed = []

        def handler(batch):
            self.processed.append((batch.device_id, len(batch.work_units)))
            time.sleep(0.01)
            return {"elapsed": 0.01, "device_id": batch.device_id}

        self.cluster = SyntheticAdapterCluster(devices=self.devices, work_handler=handler)

    def tearDown(self) -> None:
        self.cluster.shutdown()

    def test_processing_all_units(self) -> None:
        work_units = [WorkUnit(job_id="job", payload={"i": i}) for i in range(8)]
        self.cluster.submit_work_units(work_units)

        for _ in range(10):
            self.cluster.step()
            if not self.cluster.work_queue and not self.cluster._futures:
                break
            time.sleep(0.01)

        self.assertFalse(self.cluster.work_queue)
        self.assertFalse(self.cluster._futures)
        self.assertEqual(sum(count for _, count in self.processed), 8)

    def test_gpu_receives_work(self) -> None:
        work_units = [WorkUnit(job_id="job", payload={"i": i}) for i in range(8)]
        self.cluster.submit_work_units(work_units)
        
        for _ in range(10):
            self.cluster.step()
            time.sleep(0.02)

        device_ids = [device_id for device_id, _ in self.processed]
        self.assertIn("gpu", device_ids)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
