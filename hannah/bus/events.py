"""Typed bus messages for Hannah CLI ingress and egress."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, ClassVar, Mapping
from uuid import uuid4


def _new_message_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class BusMessage:
    """Base transport message for the CLI message bus."""

    channel: str
    session_id: str
    content: str
    message_id: str = field(default_factory=_new_message_id)
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    direction: ClassVar[str] = "bus"
    role: ClassVar[str] = "message"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "role": self.role,
            "channel": self.channel,
            "session_id": self.session_id,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class InboundMessage(BusMessage):
    """A user- or channel-originated message entering the bus."""

    direction: ClassVar[str] = "inbound"
    role: ClassVar[str] = "user"

    @classmethod
    def create(
        cls,
        *,
        channel: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "InboundMessage":
        return cls(channel=channel, session_id=session_id, content=content, metadata=metadata or {})


@dataclass(frozen=True, slots=True)
class OutboundMessage(BusMessage):
    """An assistant response leaving the runtime and returning to the bus."""

    direction: ClassVar[str] = "outbound"
    role: ClassVar[str] = "assistant"

    @classmethod
    def create(
        cls,
        *,
        channel: str,
        session_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OutboundMessage":
        return cls(channel=channel, session_id=session_id, content=content, metadata=metadata or {})
