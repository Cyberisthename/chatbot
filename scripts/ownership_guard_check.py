#!/usr/bin/env python3
"""Ownership hardening guard.

Fails if forbidden demo/mock/third-party wrapper paths exist in the repo.
Usage:
  python3 scripts/ownership_guard_check.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_GLOBS = [
    "demos",
    "demo_multiverse",
    "ollama-jarvis-setup",
    "docs/ollama",
    "scripts/demo_*.py",
    "scripts/*_demo*.py",
    "quantacap/.adapters/*.demo*.json",
]


def main() -> int:
    violations: list[Path] = []

    for pattern in FORBIDDEN_GLOBS:
        for match in REPO_ROOT.glob(pattern):
            if match.exists():
                violations.append(match)

    if violations:
        print("❌ Ownership guard failed. Forbidden paths detected:")
        for v in sorted(set(violations)):
            print(f" - {v.relative_to(REPO_ROOT)}")
        return 1

    print("✅ Ownership guard passed. No forbidden demo/mock/third-party wrapper paths detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
