"""OpenPitWall corpus loader with lightweight parsing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusRecord:
    """Normalized training record for downstream smoke trainers."""

    source: str
    record_id: str
    payload: dict


def _load_json_file(file_path: Path) -> list[CorpusRecord]:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    else:
        rows = []
    return [
        CorpusRecord(source=str(file_path), record_id=f"{file_path.stem}_{index}", payload=row)
        for index, row in enumerate(rows)
    ]


def _load_jsonl_file(file_path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(
                CorpusRecord(
                    source=str(file_path),
                    record_id=f"{file_path.stem}_{index}",
                    payload=payload,
                )
            )
    return records


def _load_csv_file(file_path: Path) -> list[CorpusRecord]:
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
    except Exception:
        return []
    return [
        CorpusRecord(source=str(file_path), record_id=f"{file_path.stem}_{index}", payload=row)
        for index, row in enumerate(rows)
    ]


def _load_text_file(file_path: Path) -> list[CorpusRecord]:
    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        CorpusRecord(
            source=str(file_path),
            record_id=f"{file_path.stem}_{index}",
            payload={"text": line},
        )
        for index, line in enumerate(lines)
    ]


def load_training_corpus(path: str | Path = "data/openpitwall") -> list[dict]:
    """Load OpenPitWall-derived training examples when available."""
    corpus_dir = Path(path)
    if not corpus_dir.exists():
        return []

    records: list[CorpusRecord] = []
    for file_path in sorted(file_path for file_path in corpus_dir.glob("**/*") if file_path.is_file()):
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            records.extend(_load_json_file(file_path))
        elif suffix == ".jsonl":
            records.extend(_load_jsonl_file(file_path))
        elif suffix == ".csv":
            records.extend(_load_csv_file(file_path))
        elif suffix in {".txt", ".md"}:
            records.extend(_load_text_file(file_path))
        else:
            records.append(
                CorpusRecord(
                    source=str(file_path),
                    record_id=f"{file_path.stem}_0",
                    payload={"path": str(file_path)},
                )
            )
    return [{"source": record.source, "id": record.record_id, **record.payload} for record in records]
