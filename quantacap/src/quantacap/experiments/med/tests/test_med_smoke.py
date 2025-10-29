from __future__ import annotations

import json
from pathlib import Path

from quantacap.experiments.med.docking import run_search


def test_med_search_smoke(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    result = run_search("ACE2", cycles=10, topk=3, seed=1234, adapter_id="med.test")
    assert result["candidates"], "expected non-empty candidate list"
    assert result["delta_v"] is not None
    payload = json.loads(Path("artifacts/med_ACE2_candidates.json").read_text())
    assert payload["candidates"]
