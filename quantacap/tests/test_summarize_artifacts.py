from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_quantum_artifacts import summarize_quantum_artifacts


def test_summarize_quantum_artifacts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    chsh_payload = {"S": 2.6, "terms": {"AB": 0.5, "AB'": 0.4}, "shots_per_setting": 1000}
    (artifacts_dir / "chsh_clean.json").write_text(json.dumps(chsh_payload))

    atom_payload = {"state": {"probs": [0.25, 0.75]}}
    (artifacts_dir / "atom_demo_replay.json").write_text(json.dumps(atom_payload))

    quion_payload = {
        "frames": [{"t": 0}, {"t": 1}, {"t": 2}],
        "metrics": {"F_final": 0.995, "frames_committed": 3, "ΔV_violations": 0},
    }
    (artifacts_dir / "quion_state_series.json").write_text(json.dumps(quion_payload))

    zip_path = tmp_path / "quion_experiment.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy.json", json.dumps({"dummy": True}))

    summary_path = artifacts_dir / "summary_results.json"

    summary = summarize_quantum_artifacts(zip_path, artifacts_dir, summary_path)

    assert summary_path.exists()
    assert summary["chsh"]["S"] == pytest.approx(2.6)
    assert summary["atom"]["mean"] == pytest.approx(0.5)
    assert summary["atom"]["variance"] == pytest.approx(0.0625)
    assert summary["quion"]["frame_count"] == pytest.approx(3)
    assert summary["quion"]["F_final"] == pytest.approx(0.995)


