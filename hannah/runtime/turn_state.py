from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4


@dataclass(slots=True)
class TurnState:
    session_id: str
    message_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def append_message(self, message: Mapping[str, Any]) -> None:
        self.messages.append(dict(message))

    def extend_messages(self, messages: Sequence[Mapping[str, Any]]) -> None:
        for message in messages:
            self.append_message(message)

    def snapshot_messages(self) -> list[dict[str, Any]]:
        return deepcopy(self.messages)

    def latest_user_content(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""
