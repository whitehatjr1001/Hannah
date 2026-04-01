"""Atomic artifact write helpers for model trainers."""

from __future__ import annotations

import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def atomic_pickle_dump(path: str | Path, payload: Any) -> str:
    """Write a pickle artifact atomically via temp file + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(target)


def load_pickle_artifact(path: str | Path) -> Any:
    """Load a pickle artifact from disk."""
    target = Path(path)
    with target.open("rb") as handle:
        return pickle.load(handle)


def atomic_zip_bundle(path: str | Path, entries: dict[str, str]) -> str:
    """Write a zip artifact atomically via temp file + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(target)
    try:
        with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in entries.items():
                archive.writestr(name, contents)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(target)


def _temporary_path(target: Path) -> Path:
    file_descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(file_descriptor)
    return Path(raw_path)
