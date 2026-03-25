import asyncio

import pytest

from hannah.runtime.bus import AsyncEventBus
from hannah.runtime.events import EVENT_TYPES, EventEnvelope

EXPECTED_EVENT_TYPES = {
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
}


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


@pytest.mark.anyio
async def test_event_bus_prevents_payload_mutation_from_leaking_to_later_subscribers():
    bus = AsyncEventBus()
    observed_payloads = []
    envelope = EventEnvelope.create(
        "tool_call_started",
        "session",
        "msg4",
        payload={"status": "original"},
    )

    async def mutating_handler(event):
        try:
            event.payload["status"] = "mutated"
        except TypeError:
            pass

    async def observing_handler(event):
        observed_payloads.append(dict(event.payload))

    bus.subscribe(mutating_handler)
    bus.subscribe(observing_handler)

    await bus.publish(envelope)

    assert dict(envelope.payload) == {"status": "original"}
    assert observed_payloads == [{"status": "original"}]


@pytest.mark.anyio
async def test_event_bus_serializes_concurrent_publish_calls_fifo():
    bus = AsyncEventBus()
    slow_handler_started = asyncio.Event()
    release_slow_handler = asyncio.Event()
    observed = []

    async def slow_handler(event):
        if event.message_id == "msg1":
            observed.append(("slow", "msg1", "start"))
            slow_handler_started.set()
            await release_slow_handler.wait()
            observed.append(("slow", "msg1", "end"))
            return
        observed.append(("slow", event.message_id, "end"))

    async def fast_handler(event):
        observed.append(("fast", event.message_id, "seen"))

    bus.subscribe(slow_handler)
    bus.subscribe(fast_handler)

    first_publish = asyncio.create_task(
        bus.publish(EventEnvelope.create("user_message_received", "session", "msg1"))
    )
    await slow_handler_started.wait()

    second_publish = asyncio.create_task(
        bus.publish(EventEnvelope.create("provider_request_started", "session", "msg2"))
    )
    await asyncio.sleep(0)

    assert [entry for entry in observed if entry[1] == "msg2"] == []

    release_slow_handler.set()
    await asyncio.gather(first_publish, second_publish)

    first_msg2_index = min(
        index for index, entry in enumerate(observed) if entry[1] == "msg2"
    )
    assert observed.index(("slow", "msg1", "end")) < first_msg2_index


def test_event_types_match_literal_runtime_contract():
    assert EVENT_TYPES == EXPECTED_EVENT_TYPES

    for event_type in EXPECTED_EVENT_TYPES:
        envelope = EventEnvelope.create(event_type, "session", "msg")
        assert envelope.event_type == event_type

    with pytest.raises(ValueError):
        EventEnvelope.create("unknown_event", "session", "msg")
