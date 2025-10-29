from __future__ import annotations

import json
from pathlib import Path

from quantacap.experiments.med.docking import run_search
from quantacap.experiments.pi.entropy import run_pi_entropy_control


def test_med_search_smoke(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    result = run_search("ACE2", cycles=10, topk=3, seed=1234, adapter_id="med.test")
    assert result["candidates"], "expected non-empty candidate list"
    assert result["delta_v"] is not None
    payload = json.loads(Path("artifacts/med_ACE2_candidates.json").read_text())
    assert payload["candidates"]


def test_med_search_with_pi_adapter() -> None:
    run_pi_entropy_control(steps=20, seed=55, adapter_id="pi.entropy.test")
    result = run_search(
        "ACE2",
        cycles=10,
        topk=2,
        seed=4321,
        adapter_id="med.phase",
        pi_adapter="pi.entropy.test",
    )
    assert "phase_weight" in result
    assert 0.0 < result["phase_weight"] <= 1.0
