"""Project-wide site customizations.

Ensures the local ``src`` directory is importable without installing the
package. This makes ``python -m atomsim.cli`` work out-of-the-box.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.exists():
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
