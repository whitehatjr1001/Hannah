import asyncio

import pytest

from hannah.runtime.bus import AsyncEventBus
from hannah.runtime.events import EVENT_TYPES, EventEnvelope


@pytest.mark.anyio
async def test_event_bus_delivers_events_in_order():
    bus = AsyncEventBus()
    seen = []

    async def handler(event):
        await asyncio.sleep(0)
        seen.append(event.event_type)

    bus.subscribe(handler)
    await bus.publish(EventEnvelope.create("user_message_received", "session", "msg1"))
    await bus.publish(EventEnvelope.create("provider_request_started", "session", "msg2"))

    assert seen == ["user_message_received", "provider_request_started"]


@pytest.mark.anyio
async def test_event_bus_isolates_failing_subscribers():
    bus = AsyncEventBus()
    captured = []

    async def exploding_handler(event):
        raise RuntimeError("boom")

    async def resilient_handler(event):
        captured.append(event.message_id)

    bus.subscribe(exploding_handler)
    bus.subscribe(resilient_handler)

    await bus.publish(EventEnvelope.create("tool_call_started", "session", "msg3"))
    assert captured == ["msg3"]


def test_event_envelope_requires_known_type():
    for event_type in EVENT_TYPES:
        envelope = EventEnvelope.create(event_type, "session", "msg")
        assert envelope.event_type == event_type

    with pytest.raises(ValueError):
        EventEnvelope.create("unknown_event", "session", "msg")
