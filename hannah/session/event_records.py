"""JSONL-safe event record helpers for persisted session streams."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from hannah.runtime.events import EventEnvelope


def serialize_event_record(event: EventEnvelope, *, session_id: str) -> dict[str, Any]:
    return {
        "record_type": "event",
        "session_id": session_id,
        "created_at": event.timestamp.isoformat(),
        "payload": {
            "event_type": event.event_type,
            "message_id": event.message_id,
            "worker_id": event.worker_id,
            "payload": _json_safe_mapping(event.payload),
        },
    }


def is_event_record(payload: Mapping[str, Any]) -> bool:
    return payload.get("record_type") == "event"


def _json_safe_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in payload.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
