from __future__ import annotations

from datetime import datetime

from hannah.bus.events import InboundMessage, OutboundMessage


def test_bus_messages_capture_direction_role_and_metadata() -> None:
    metadata = {"source": "chat", "priority": "high"}
    inbound = InboundMessage.create(
        channel="cli",
        session_id="session-1",
        content="hello",
        metadata=metadata,
    )
    outbound = OutboundMessage.create(
        channel="cli",
        session_id="session-1",
        content="box this lap",
    )

    metadata["source"] = "mutated"

    inbound_payload = inbound.to_dict()
    outbound_payload = outbound.to_dict()

    assert inbound.direction == "inbound"
    assert inbound.role == "user"
    assert outbound.direction == "outbound"
    assert outbound.role == "assistant"
    assert inbound_payload["metadata"] == {"source": "chat", "priority": "high"}
    assert outbound_payload["metadata"] == {}
    assert datetime.fromisoformat(inbound_payload["timestamp"])
    assert datetime.fromisoformat(outbound_payload["timestamp"])


def test_bus_messages_are_distinct_types_with_stable_ids() -> None:
    first = InboundMessage.create(channel="cli", session_id="s", content="first")
    second = InboundMessage.create(channel="cli", session_id="s", content="second")

    assert type(first) is InboundMessage
    assert type(second) is InboundMessage
    assert first.message_id != second.message_id
    assert first.to_dict()["session_id"] == "s"
