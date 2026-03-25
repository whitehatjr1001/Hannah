from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List, Optional

from rich.console import Console

from .events import EVENT_TYPES, EventEnvelope

Subscriber = Callable[[EventEnvelope], Awaitable[None]]
CONSOLE = Console()


class AsyncEventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[Optional[str], List[Subscriber]] = defaultdict(list)
        self._locks: Dict[Subscriber, asyncio.Lock] = {}
        self._publish_lock = asyncio.Lock()

    def subscribe(
        self, handler: Subscriber, event_type: Optional[str] = None
    ) -> None:
        if event_type is not None and event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event type for subscription: {event_type}")

        self._subscribers[event_type].append(handler)

    async def publish(self, envelope: EventEnvelope) -> None:
        if envelope.event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {envelope.event_type}")

        async with self._publish_lock:
            tasks: List[Awaitable[None]] = []
            for subscribed_type, handlers in self._subscribers.items():
                if subscribed_type is not None and subscribed_type != envelope.event_type:
                    continue
                for handler in handlers:
                    tasks.append(self._dispatch(handler, envelope))

            if not tasks:
                return

            await asyncio.gather(*tasks)

    async def _dispatch(self, handler: Subscriber, envelope: EventEnvelope) -> None:
        lock = self._locks.setdefault(handler, asyncio.Lock())
        async with lock:
            try:
                await handler(envelope)
            except Exception as exc:  # pragma: no cover - logging path
                CONSOLE.print(
                    f"[yellow]Subscriber error for {envelope.event_type}: {exc}[/yellow]"
                )
