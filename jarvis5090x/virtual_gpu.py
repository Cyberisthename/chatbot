from __future__ import annotations

from typing import Any, Dict

from .orchestrator import Jarvis5090X


class VirtualGPU:
    """Convenient facade for interacting with the Jarvis-5090X orchestrator."""

    def __init__(self, orchestrator: Jarvis5090X) -> None:
        self.orchestrator = orchestrator

    def submit(self, op_type: str, op_signature: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.submit(op_type, op_signature, payload)

    def benchmark(self, op_type: str, op_signature: str, payload: Dict[str, Any], repeat: int = 3) -> Dict[str, Any]:
        results = []
        for idx in range(repeat):
            signature = f"{op_signature}::run{idx}"
            results.append(self.orchestrator.submit(op_type, signature, payload))
        stats = self.orchestrator.benchmark_stats()
        return {"results": results, "stats": stats}
