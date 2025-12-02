import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


def safe_json_load(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def checksum(path: str, algo: str = "sha256") -> Optional[str]:
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def list_adapters(root: str = ".adapters") -> List[str]:
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in os.listdir(root):
        if name.endswith(".json"):
            out.append(os.path.join(root, name))
    return sorted(out)


def git_info() -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(args, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None

    status = _run(["git", "status", "--porcelain"])
    if status and len(status) > 2000:
        status = status[:2000] + "…"

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": status,
    }


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "time": time.time(),
    }
    try:
        import numpy as np  # type: ignore

        info["numpy"] = {"version": np.__version__}
    except Exception:
        info["numpy"] = None
    try:
        import cupy as cp  # type: ignore

        info["cupy"] = {"available": True, "version": cp.__version__}
    except Exception:
        info["cupy"] = {"available": False}
    return info


def md_kv(title: str, kv: Dict[str, Any]) -> str:
    lines = [f"## {title}"]
    for key, value in kv.items():
        lines.append(f"- **{key}**: `{value}`")
    return "\n".join(lines)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def glob_first(*patterns: str) -> List[str]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    return sorted(set(matches))
