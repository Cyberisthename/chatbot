from __future__ import annotations

import argparse
import json
import os
import sys
from math import fsum
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._report_utils import safe_json_load


def _extract_zip(zip_path: Path, target_dir: Path) -> List[str]:
    if not zip_path.exists():
        return []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
            return [Path(name).name for name in zf.namelist() if name.endswith(".json")]
    except zipfile.BadZipFile:
        return []


def _scan_json_files(root: Path) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for path in sorted(root.rglob("*.json")):
        rel_name = path.name
        data = safe_json_load(str(path))
        if data is None:
            report[rel_name] = "Non-JSON or binary data"
        else:
            report[rel_name] = data
    return report


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return fsum(values) / len(values)


def _variance(values: Sequence[float], mean_value: float | None = None) -> float:
    if not values:
        return 0.0
    mu = mean_value if mean_value is not None else _mean(values)
    return fsum((v - mu) ** 2 for v in values) / len(values)


def _to_float_list(values: Iterable[object]) -> List[float]:
    numbers: List[float] = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return numbers


def _summary_from_chsh(data: Dict[str, object]) -> Dict[str, float]:
    if not data:
        return {}
    summary: Dict[str, float] = {}
    for key in ("S", "S_value", "S_clean"):
        if key in data:
            try:
                summary["S"] = float(data[key])
            except (TypeError, ValueError):
                pass
            break
    if "terms" in data and isinstance(data["terms"], dict):
        terms_values = _to_float_list(data["terms"].values())
        if terms_values:
            summary["terms_mean"] = _mean(terms_values)
    if "shots" in data:
        try:
            summary["shots"] = float(data["shots"])
        except (TypeError, ValueError):
            pass
    elif "shots_per_setting" in data:
        try:
            summary["shots_per_setting"] = float(data["shots_per_setting"])
        except (TypeError, ValueError):
            pass
    return summary


def _extract_probs(record: Dict[str, object]) -> List[float]:
    if not record:
        return []
    if isinstance(record, dict) and "probabilities" in record:
        return _to_float_list(record.get("probabilities", []))
    state = record.get("state") if isinstance(record, dict) else None
    if isinstance(state, dict) and "probs" in state:
        return _to_float_list(state.get("probs", []))
    return []


def _summary_from_atom(record: Dict[str, object]) -> Dict[str, float]:
    probs = _extract_probs(record)
    if not probs:
        return {}
    mean_value = _mean(probs)
    variance_value = _variance(probs, mean_value)
    return {
        "mean": mean_value,
        "variance": variance_value,
        "max": max(probs),
        "min": min(probs),
    }


def _summary_from_quion(record: Dict[str, object]) -> Dict[str, float]:
    if not record:
        return {}
    frames: Iterable[Dict[str, object]] = record.get("frames", []) if isinstance(record, dict) else []
    frames_list = list(frames)
    if not frames_list:
        return {}
    metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
    summary: Dict[str, float] = {
        "frame_count": float(len(frames_list)),
    }
    if isinstance(metrics, dict):
        for key, dest in (
            ("F_final", "F_final"),
            ("frames_committed", "frames_committed"),
            ("ΔV_violations", "delta_v_violations"),
        ):
            if key in metrics:
                try:
                    summary[dest] = float(metrics[key])
                except (TypeError, ValueError):
                    continue
    return summary


def summarize_quantum_artifacts(
    zip_path: Path,
    artifacts_dir: Path,
    summary_path: Path,
) -> Dict[str, Dict[str, float]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = _extract_zip(zip_path, artifacts_dir)

    report = _scan_json_files(artifacts_dir)
    if extracted_files:
        print(f"📦 Extracted files: {extracted_files}")
    else:
        print("📦 No zip extraction performed or archive missing/invalid")
    print(f"🔍 Detected JSON files: {sorted(report.keys())}")

    summary: Dict[str, Dict[str, float]] = {}

    chsh_record = report.get("chsh_clean.json")
    if isinstance(chsh_record, dict):
        chsh_summary = _summary_from_chsh(chsh_record)
        if chsh_summary:
            summary["chsh"] = chsh_summary

    atom_record = report.get("atom_demo_replay.json") or report.get("atom1d_atom.demo.json")
    if isinstance(atom_record, dict):
        atom_summary = _summary_from_atom(atom_record)
        if atom_summary:
            summary["atom"] = atom_summary

    quion_record = report.get("quion_state_series.json")
    if isinstance(quion_record, dict):
        quion_summary = _summary_from_quion(quion_record)
        if quion_summary:
            summary["quion"] = quion_summary

    output_payload: Dict[str, object] = dict(summary)

    chsh_summary = summary.get("chsh", {})
    atom_summary = summary.get("atom", {})

    if isinstance(chsh_summary, dict) and "S" in chsh_summary:
        output_payload["chsh_S_value"] = chsh_summary.get("S")

    if isinstance(atom_summary, dict):
        if "mean" in atom_summary:
            output_payload["atom_mean"] = atom_summary.get("mean")
        if "variance" in atom_summary:
            output_payload["atom_variance"] = atom_summary.get("variance")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, sort_keys=True)
    print(f"✅ Summaries saved to {summary_path}")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize quantum artifacts")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        default="quion_experiment.zip",
        help="Path to the quion experiment zip archive",
    )
    parser.add_argument(
        "--artifacts",
        dest="artifacts_dir",
        default="artifacts",
        help="Directory containing extracted artifact files",
    )
    parser.add_argument(
        "--output",
        dest="summary_path",
        default=os.path.join("artifacts", "summary_results.json"),
        help="Where to write the summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    zip_path = Path(args.zip_path)
    artifacts_dir = Path(args.artifacts_dir)
    summary_path = Path(args.summary_path)

    if not zip_path.exists():
        fallback = artifacts_dir / zip_path.name
        if fallback.exists():
            zip_path = fallback

    summarize_quantum_artifacts(zip_path, artifacts_dir, summary_path)


if __name__ == "__main__":
    main()
