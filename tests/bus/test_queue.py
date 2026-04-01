from __future__ import annotations

import asyncio

import pytest

from hannah.bus.events import InboundMessage, OutboundMessage
from hannah.bus.queue import MessageBus, MessageQueue, run_bus_turn


def test_message_queue_preserves_fifo_order() -> None:
    queue: MessageQueue[InboundMessage] = MessageQueue("inbound")
    first = InboundMessage.create(channel="cli", session_id="s", content="first")
    second = InboundMessage.create(channel="cli", session_id="s", content="second")

    async def _exercise() -> None:
        await queue.put(first)
        await queue.put(second)

        assert queue.qsize() == 2
        assert await queue.get() == first
        assert queue.drain() == [second]

    asyncio.run(_exercise())


def test_message_bus_routes_messages_by_direction() -> None:
    bus = MessageBus()
    inbound = InboundMessage.create(channel="cli", session_id="s", content="hello")
    outbound = OutboundMessage.create(channel="cli", session_id="s", content="reply")

    async def _exercise() -> None:
        await bus.publish(inbound)
        await bus.publish(outbound)

        assert bus.inbound.qsize() == 1
        assert bus.outbound.qsize() == 1
        assert await bus.receive_inbound() == inbound
        assert await bus.receive_outbound() == outbound

    asyncio.run(_exercise())


def test_run_bus_turn_publishes_inbound_and_outbound_messages() -> None:
    class _FakeLoop:
        async def run_turn(self, user_input: str, *, session_id: str = "default") -> str:
            return f"{session_id}:{user_input.upper()}"

    bus = MessageBus()

    async def _exercise() -> None:
        outbound = await run_bus_turn(
            agent_loop=_FakeLoop(),
            message="should we pit",
            session_id="cli:test",
            channel="chat",
            bus=bus,
        )

        assert outbound.content == "cli:test:SHOULD WE PIT"
        assert bus.drain_inbound() == []
        assert [message.content for message in bus.drain_outbound()] == ["cli:test:SHOULD WE PIT"]

    asyncio.run(_exercise())
