import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._report_utils import (
    checksum,
    env_info,
    git_info,
    glob_first,
    list_adapters,
    md_kv,
    safe_json_load,
    write_json,
)

ART = "artifacts"
ADP = ".adapters"

GUESSES = {
    "chsh_scan": [
        f"{ART}/chsh_clean.json",
        f"{ART}/chsh_scan_replay.json",
        f"{ART}/chsh.scan.demo*.json",
        f"/tmp/chsh_scan*.json",
    ],
    "chsh_y": [f"{ART}/chsh_y*.json", f"/tmp/chsh_y*.json"],
    "bell": [f"{ART}/bell_clean.json", f"{ART}/bell_depol_*.json"],
    "fringe": [f"{ART}/fringe_clean.json"],
    "atom1d": [f"{ART}/atom1d_*.json", f"{ART}/atom_demo_replay.json"],
    "grover": [f"{ART}/grover_*.json"],
}


def summarize_chsh_scan(obj: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not obj:
        return None
    out: Dict[str, Any] = {}
    for key in ("S_clean", "p", "S_noisy", "S_rehearsed"):
        if key in obj:
            out[key] = obj[key]
    if "S_rehearsed" in out and "p" in out:
        arr = out["S_rehearsed"]
        ps = out["p"]
        if isinstance(arr, list) and isinstance(ps, list) and len(arr) == len(ps) and arr:
            peak_index = max(range(len(arr)), key=lambda idx: arr[idx])
            out["peak"] = {"p_star": ps[peak_index], "S": arr[peak_index]}
    return out or None


def symmetry_error(density: Any) -> float | None:
    if not isinstance(density, list):
        return None
    n = len(density)
    if n == 0:
        return None
    m = n // 2
    if m == 0:
        return 0.0
    err = 0.0
    for i in range(m):
        try:
            a = float(density[i])
            b = float(density[-(i + 1)])
        except (TypeError, ValueError):
            return None
        err += abs(a - b)
    return err / m


def summarize_atom(obj: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not obj:
        return None
    density = obj.get("density") or obj.get("trace", {}).get("density")
    grid = obj.get("grid_x") or obj.get("trace", {}).get("grid_x")
    out: Dict[str, Any] = {"points": len(density) if isinstance(density, list) else 0}
    if isinstance(density, list) and density:
        total = float(sum(density))
        out["norm_sum"] = total
        out["sym_err"] = symmetry_error(density)
    if isinstance(grid, list) and grid:
        out["x_min"] = min(grid)
        out["x_max"] = max(grid)
    return out


def first_existing(patterns: list[str]) -> str | None:
    for pattern in patterns:
        for path in glob_first(pattern):
            if os.path.isfile(path):
                return path
    return None


def main() -> None:
    os.makedirs(ART, exist_ok=True)

    env = env_info()
    git = git_info()

    artifacts: Dict[str, Dict[str, Any]] = {}
    for key, patterns in GUESSES.items():
        path = first_existing(patterns)
        if not path:
            continue
        payload = safe_json_load(path)
        artifacts[key] = {
            "path": path,
            "checksum": checksum(path),
            "summary": None,
            "raw_excerpt": None,
        }
        if key == "chsh_scan":
            artifacts[key]["summary"] = summarize_chsh_scan(payload)
        elif key == "atom1d":
            artifacts[key]["summary"] = summarize_atom(payload)
        else:
            if isinstance(payload, dict):
                excerpt = {}
                for idx, k in enumerate(payload.keys()):
                    if idx >= 6:
                        break
                    excerpt[k] = payload.get(k)
                artifacts[key]["raw_excerpt"] = excerpt

    adapter_files = list_adapters(ADP)
    adapters_meta: list[Dict[str, Any]] = []
    cap = 50
    for path in adapter_files[:cap]:
        record = safe_json_load(path) or {}
        adapters_meta.append(
            {
                "path": path,
                "checksum": checksum(path),
                "keys": list(record.keys()),
                "id": record.get("id"),
                "ts": record.get("ts"),
                "data_keys": list((record.get("data") or {}).keys()),
            }
        )

    report = {
        "meta": {
            "title": "Quantacap Discovery Report",
            "timestamp": env.get("time"),
            "notes": "Synthetic quantum results; not physical measurements.",
        },
        "env": env,
        "git": git,
        "artifacts": artifacts,
        "adapters": {
            "count": len(adapter_files),
            "listed": adapters_meta,
            "truncated": max(0, len(adapter_files) - len(adapters_meta)),
        },
    }

    json_path = f"{ART}/discovery_report.json"
    write_json(json_path, report)

    md_lines = ["# Quantacap Discovery Report\n"]
    md_lines.append(
        md_kv(
            "Environment",
            {
                "python": (report["env"]["python"].split()[0] if report.get("env") else None),
                "platform": report["env"].get("platform") if report.get("env") else None,
                "numpy": (report["env"].get("numpy") or {}).get("version") if report.get("env") else None,
                "cupy.available": (report["env"].get("cupy") or {}).get("available") if report.get("env") else None,
                "git.commit": report["git"].get("commit"),
                "git.branch": report["git"].get("branch"),
            },
        )
    )

    if "chsh_scan" in artifacts and artifacts["chsh_scan"].get("summary"):
        summary = artifacts["chsh_scan"]["summary"]
        md_lines.append("\n## CHSH Noise/Rehearsal Summary")
        md_lines.append(f"- S_clean: `{summary.get('S_clean')}`")
        peak = summary.get("peak") if isinstance(summary, dict) else None
        if isinstance(peak, dict):
            md_lines.append(f"- Rehearsed Peak: p*=`{peak.get('p_star')}`, S=`{peak.get('S')}`")

    if "chsh_y" in artifacts:
        md_lines.append("\n## CHSH-Y (Y/G Modulated) — Artifact Present")
        md_lines.append(f"- File: `{artifacts['chsh_y']['path']}`")

    if "atom1d" in artifacts and artifacts["atom1d"].get("summary"):
        atom_summary = artifacts["atom1d"]["summary"]
        md_lines.append("\n## Atom-1D Summary")
        md_lines.append(f"- Points: `{atom_summary.get('points')}`")
        md_lines.append(f"- Normalization sum: `{atom_summary.get('norm_sum')}`")
        md_lines.append(f"- Symmetry error: `{atom_summary.get('sym_err')}`")
        md_lines.append(f"- x-range: `[{atom_summary.get('x_min')}, {atom_summary.get('x_max')}]`")

    md_lines.append(
        f"\n## Adapters\n- Total: `{report['adapters']['count']}` (showing {len(adapters_meta)}, truncated {report['adapters']['truncated']})"
    )
    md_lines.append(f"- Report JSON: `{json_path}`\n")

    md_path = f"{ART}/discovery_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    zip_path = f"{ART}/discovery_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(json_path, arcname="discovery_report.json")
        bundle.write(md_path, arcname="discovery_report.md")
        for meta in artifacts.values():
            p = meta.get("path")
            if p and os.path.isfile(p):
                bundle.write(p, arcname=os.path.basename(p))
        for path in adapter_files[:cap]:
            if os.path.isfile(path):
                bundle.write(path, arcname=f"adapters/{os.path.basename(path)}")

    result = {
        "ok": True,
        "json": json_path,
        "markdown": md_path,
        "zip": zip_path,
        "artifacts_found": [key for key, value in artifacts.items() if value.get("path")],
        "adapters_included": min(len(adapter_files), cap),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
