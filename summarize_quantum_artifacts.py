#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent
QUANTACAP_DIR = ROOT / "quantacap"
if str(QUANTACAP_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTACAP_DIR))

from scripts.summarize_quantum_artifacts import summarize_quantum_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize quantum artifacts from the Quion experiment bundle")
    parser.add_argument("--zip", default="quion_experiment.zip", help="Path to the artifacts zip archive")
    parser.add_argument("--artifacts", default="artifacts", help="Directory to extract or locate artifacts")
    parser.add_argument(
        "--output",
        default="artifacts/summary_results.json",
        help="Where to write the summary JSON file",
    )
    parser.add_argument("--print", dest="print_summary", action="store_true", help="Print the flattened summary to stdout")
    return parser.parse_args()


def _flatten(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    chsh = summary.get("chsh") or {}
    atom = summary.get("atom") or {}

    if isinstance(chsh, dict) and "S" in chsh:
        try:
            flat["chsh_S_value"] = float(chsh["S"])
        except (TypeError, ValueError):
            pass

    if isinstance(atom, dict):
        for key in ("mean", "variance"):
            if key in atom:
                try:
                    flat[f"atom_{key}"] = float(atom[key])
                except (TypeError, ValueError):
                    continue

    return flat


def main() -> None:
    args = _parse_args()

    zip_path = Path(args.zip)
    artifacts_dir = Path(args.artifacts)
    summary_path = Path(args.output)

    if not zip_path.exists():
        fallback = artifacts_dir / zip_path.name
        if fallback.exists():
            zip_path = fallback

    summary = summarize_quantum_artifacts(zip_path, artifacts_dir, summary_path)

    if args.print_summary:
        flat = _flatten(summary)
        if flat:
            print(json.dumps(flat, indent=2))
        else:
            print("{}")


if __name__ == "__main__":
    main()
