"""Async message queues used by the Hannah CLI bus."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from hannah.bus.events import BusMessage, InboundMessage, OutboundMessage

TMessage = TypeVar("TMessage", bound=BusMessage)


@dataclass(slots=True)
class MessageQueue(Generic[TMessage]):
    """Thin FIFO wrapper around ``asyncio.Queue``."""

    name: str
    _queue: asyncio.Queue[TMessage] = field(default_factory=asyncio.Queue, init=False, repr=False)

    async def put(self, message: TMessage) -> None:
        await self._queue.put(message)

    def put_nowait(self, message: TMessage) -> None:
        self._queue.put_nowait(message)

    async def get(self) -> TMessage:
        return await self._queue.get()

    def get_nowait(self) -> TMessage:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def drain(self) -> list[TMessage]:
        drained: list[TMessage] = []
        while not self._queue.empty():
            drained.append(self._queue.get_nowait())
        return drained


@dataclass(slots=True)
class MessageBus:
    """Bidirectional message bus with separate inbound and outbound queues."""

    inbound: MessageQueue[InboundMessage] = field(default_factory=lambda: MessageQueue("inbound"))
    outbound: MessageQueue[OutboundMessage] = field(default_factory=lambda: MessageQueue("outbound"))

    async def publish(self, message: BusMessage) -> None:
        queue = self._queue_for(message)
        await queue.put(message)  # type: ignore[arg-type]

    async def send_inbound(self, message: InboundMessage) -> None:
        await self.inbound.put(message)

    async def send_outbound(self, message: OutboundMessage) -> None:
        await self.outbound.put(message)

    async def receive_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def receive_outbound(self) -> OutboundMessage:
        return await self.outbound.get()

    def drain_inbound(self) -> list[InboundMessage]:
        return self.inbound.drain()

    def drain_outbound(self) -> list[OutboundMessage]:
        return self.outbound.drain()

    def _queue_for(self, message: BusMessage) -> MessageQueue[Any]:
        if getattr(message, "direction", None) == InboundMessage.direction:
            return self.inbound
        if getattr(message, "direction", None) == OutboundMessage.direction:
            return self.outbound
        raise TypeError(f"unsupported bus message direction: {getattr(message, 'direction', None)!r}")


async def run_bus_turn(
    *,
    agent_loop: Any,
    message: str,
    session_id: str,
    channel: str,
    bus: MessageBus | None = None,
) -> OutboundMessage:
    """Run a single CLI turn through the bus ingress/egress shape."""

    message_bus = bus or MessageBus()
    inbound = InboundMessage.create(channel=channel, session_id=session_id, content=message)
    await message_bus.publish(inbound)
    received = await message_bus.receive_inbound()

    run_turn = getattr(agent_loop, "run_turn", None)
    if not callable(run_turn):
        raise RuntimeError("agent loop must define run_turn()")

    if "session_id" in inspect.signature(run_turn).parameters:
        response = await run_turn(received.content, session_id=session_id)
    else:
        response = await run_turn(received.content)
    outbound = OutboundMessage.create(channel=channel, session_id=session_id, content=response)
    await message_bus.publish(outbound)
    return outbound
