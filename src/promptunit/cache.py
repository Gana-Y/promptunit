"""A tiny, git-friendly on-disk cache: one small JSON file per LLM call."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple


def make_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str):
        self.directory = directory

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, key[:2], key + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path(key), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put(self, key: str, data: Dict[str, Any]) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def info(self) -> Tuple[int, int]:
        """Return (number of entries, total bytes)."""
        count = size = 0
        for root, _dirs, files in os.walk(self.directory):
            for name in files:
                if name.endswith(".json"):
                    count += 1
                    size += os.path.getsize(os.path.join(root, name))
        return count, size

    def clear(self) -> int:
        count, _ = self.info()
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        return count
