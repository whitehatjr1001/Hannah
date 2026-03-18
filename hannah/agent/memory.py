"""SQLite-backed session memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Memory:
    """Store recent conversation messages in a local SQLite database."""

    def __init__(self, db_path: str | Path = "data/hannah_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add(self, role: str, content: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(role, content) VALUES (?, ?)",
                (role, content),
            )

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages")

