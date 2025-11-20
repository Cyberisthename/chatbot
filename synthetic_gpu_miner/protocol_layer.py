import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Dict, Optional


@dataclass(slots=True)
class MiningJob:
    job_id: str
    header_prefix: bytes
    target: int
    nonce_start: int
    nonce_end: int


class ProtocolLayer:
    def __init__(self):
        self._job_queue: Queue[MiningJob] = Queue()
        self._share_queue: Queue[Dict[str, object]] = Queue()
        self._job_callback: Optional[Callable[[MiningJob], None]] = None
        self._stop_event = threading.Event()
        self._receiver_thread = threading.Thread(target=self._job_dispatch_loop, daemon=True)
        self._receiver_thread.start()

    def register_job_callback(self, callback: Callable[[MiningJob], None]) -> None:
        self._job_callback = callback

    def submit_job(self, job: MiningJob) -> None:
        self._job_queue.put(job)

    def submit_share(self, share: Dict[str, object]) -> None:
        self._share_queue.put(share)

    def get_next_share(self, timeout: float = 0.0) -> Optional[Dict[str, object]]:
        try:
            return self._share_queue.get(timeout=timeout)
        except Empty:
            return None

    def _job_dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except Empty:
                continue
            if self._job_callback:
                self._job_callback(job)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._receiver_thread.join(timeout=2)

    def simulate_pool_submission(self, share: Dict[str, object]) -> None:
        self.submit_share(share)

    def wait_for_share(self, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            share = self.get_next_share(timeout=0.1)
            if share:
                return share
        return None
