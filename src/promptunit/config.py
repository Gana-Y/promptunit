"""Global settings. Controlled by env vars or pytest command-line options."""
from __future__ import annotations

import os
from typing import Optional

MODES = ("auto", "live", "record", "replay")

_mode: Optional[str] = None
_cache_dir: Optional[str] = None


def get_mode() -> str:
    mode = _mode or os.environ.get("PROMPTUNIT_MODE", "auto")
    if mode not in MODES:
        raise ValueError(f"Invalid PROMPTUNIT_MODE {mode!r}; choose one of {MODES}")
    return mode


def set_mode(mode: Optional[str]) -> None:
    global _mode
    if mode is not None and mode not in MODES:
        raise ValueError(f"Invalid mode {mode!r}; choose one of {MODES}")
    _mode = mode


def get_cache_dir() -> str:
    return _cache_dir or os.environ.get("PROMPTUNIT_CACHE_DIR", ".promptunit_cache")


def set_cache_dir(path: Optional[str]) -> None:
    global _cache_dir
    _cache_dir = path
