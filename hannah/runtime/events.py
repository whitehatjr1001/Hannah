from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

RUNTIME_EVENT_TYPES: Tuple[str, ...] = (
    "user_message_received",
    "provider_request_started",
    "provider_response_received",
    "tool_call_started",
    "tool_call_finished",
    "subagent_spawned",
    "subagent_progress",
    "subagent_completed",
    "final_answer_emitted",
    "error_emitted",
)

EVENT_TYPES = frozenset(RUNTIME_EVENT_TYPES)


@dataclass(frozen=True)
class EventEnvelope:
    event_type: str
    timestamp: datetime
    session_id: str
    message_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {self.event_type}")

    @classmethod
    def create(
        cls,
        event_type: str,
        session_id: str,
        message_id: str,
        payload: Optional[Mapping[str, Any]] = None,
        worker_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "EventEnvelope":
        recorded_timestamp = timestamp or datetime.now(timezone.utc)
        payload_data: Dict[str, Any] = dict(payload) if payload else {}
        return cls(
            event_type=event_type,
            session_id=session_id,
            message_id=message_id,
            payload=payload_data,
            worker_id=worker_id,
            timestamp=recorded_timestamp,
        )
