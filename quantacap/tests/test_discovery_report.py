import json
import os
import subprocess
import sys
from pathlib import Path


def test_discovery_report_smoke():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "collect_discovery.py")],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stdout + proc.stderr)

    json_path = root / "artifacts" / "discovery_report.json"
    md_path = root / "artifacts" / "discovery_report.md"
    zip_path = root / "artifacts" / "discovery_bundle.zip"
    assert json_path.is_file()
    assert md_path.is_file()
    assert zip_path.is_file()

    with json_path.open("r") as f:
        data = json.load(f)
    assert "env" in data
    assert "artifacts" in data
