"""Nanobot-inspired JSONL chat session persistence."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_sessions_dir() -> Path:
    return Path(os.getenv("HANNAH_SESSION_DIR", "data/sessions"))


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("._") or "session"


def create_session_key(channel: str = "cli") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{channel}:{stamp}"


@dataclass
class Session:
    """Append-only conversation session stored in JSONL format."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        history: list[dict[str, Any]] = []
        for message in self.messages[-n:]:
            entry = {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            }
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            history.append(entry)
        return history

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now()


class SessionManager:
    """Manage JSONL-backed chat sessions under the Hannah data directory."""

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        self.sessions_dir = Path(sessions_dir or _default_sessions_dir())
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        safe_key = _safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            created_at: datetime | None = None
            updated_at: datetime | None = None
            metadata: dict[str, Any] = {}
            messages: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line.strip()
                    if not record:
                        continue
                    payload = json.loads(record)
                    if payload.get("_type") == "metadata":
                        created_at = _parse_timestamp(payload.get("created_at"))
                        updated_at = _parse_timestamp(payload.get("updated_at"))
                        metadata = payload.get("metadata", {})
                        continue
                    messages.append(payload)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or created_at or datetime.now(),
                metadata=metadata,
            )
        except Exception:
            return None

    def save(self, session: Session) -> None:
        path = self._get_session_path(session.key)
        metadata = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "message_count": len(session.messages),
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for message in session.messages:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first_line = handle.readline().strip()
                    if not first_line:
                        continue
                    metadata = json.loads(first_line)
                    if metadata.get("_type") != "metadata":
                        continue
                    key = metadata.get("key") or path.stem.replace("_", ":", 1)
                    sessions.append(
                        {
                            "key": key,
                            "created_at": metadata.get("created_at"),
                            "updated_at": metadata.get("updated_at"),
                            "message_count": int(metadata.get("message_count", 0)),
                            "path": str(path),
                        }
                    )
            except Exception:
                continue
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


@dataclass
class SessionMemory:
    """Adapter exposing session history through the existing Memory interface."""

    manager: SessionManager
    session: Session

    def add(self, role: str, content: str) -> None:
        self.session.add_message(role, content)
        self.manager.save(self.session)

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self.session.get_recent(n)

    def clear(self) -> None:
        self.session.clear()
        self.manager.save(self.session)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
