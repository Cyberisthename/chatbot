"""Aggregate discovery artefacts into a single bundle."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

TARGET_ZIP = Path("artifacts/discovery_bundle.zip")


def _safe_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # pragma: no cover - defensive
        return None


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    report: dict[str, object] = {}

    # Medicinal candidates
    med_reports = sorted(artifacts.glob("med_*_candidates.json"))
    if med_reports:
        data = _safe_json(med_reports[-1]) or {}
        report["medicinal"] = {
            "path": str(med_reports[-1]),
            "top_score": data.get("best_score"),
            "count": len(data.get("candidates", [])),
        }

    # 3D map metadata
    compmaps = sorted(artifacts.glob("compmap*_meta.json"))
    if compmaps:
        report["compmap"] = json.loads(compmaps[-1].read_text())

    # Astro ring metadata
    astro_meta = sorted(artifacts.glob("astro_ring_meta.json"))
    if astro_meta:
        report["astro"] = json.loads(astro_meta[-1].read_text())

    # π-phase extensions
    pi_artifacts = sorted(artifacts.glob("pi_*"))
    report["pi_artifacts"] = [str(path) for path in pi_artifacts if path.suffix == ".json"]

    with (artifacts / "discovery_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in report.get("pi_artifacts", []):
            zf.write(item, arcname=os.path.basename(item))
        if med_reports:
            zf.write(med_reports[-1], arcname=os.path.basename(med_reports[-1]))
        if compmaps:
            zf.write(compmaps[-1], arcname=os.path.basename(compmaps[-1]))
        if astro_meta:
            zf.write(astro_meta[-1], arcname=os.path.basename(astro_meta[-1]))
        zf.write(artifacts / "discovery_report.json", arcname="discovery_report.json")

    print(json.dumps({"bundle": str(TARGET_ZIP)}, indent=2))


if __name__ == "__main__":
    main()
