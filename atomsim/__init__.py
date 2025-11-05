"""Shim package that points to the real implementation under ``src/atomsim``."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "atomsim"
_src_init = _src_pkg / "__init__.py"

_spec = importlib.util.spec_from_file_location(
    __name__,
    _src_init,
    submodule_search_locations=[str(_src_pkg)],
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to locate src/atomsim package")

_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
