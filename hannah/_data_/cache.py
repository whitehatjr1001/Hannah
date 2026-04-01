"""Small JSON cache helpers for external API responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    """Persist JSON responses on disk."""

    def __init__(self, cache_dir: str | Path = "data/fastf1_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_path(self, prefix: str, params: dict[str, Any]) -> Path:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.md5(payload).hexdigest()
        return self.cache_dir / f"{prefix}_{digest}.json"

    def load(self, prefix: str, params: dict[str, Any]) -> Any | None:
        path = self.build_path(prefix, params)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, prefix: str, params: dict[str, Any], payload: Any) -> None:
        path = self.build_path(prefix, params)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

