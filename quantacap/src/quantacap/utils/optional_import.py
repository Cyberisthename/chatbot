"""Utilities for importing optional dependencies on demand."""

from __future__ import annotations


def optional_import(mod_name: str, *, pip_name: str | None = None, purpose: str | None = None):
    """Import *mod_name* lazily and raise a helpful error if unavailable."""
    try:
        return __import__(mod_name, fromlist=["*"])
    except ImportError as exc:
        hint = pip_name or mod_name
        detail = f" to {purpose}" if purpose else ""
        raise RuntimeError(
            f"Optional dependency '{mod_name}' is not installed{detail}. "
            f"Install via: pip install {hint}"
        ) from exc
