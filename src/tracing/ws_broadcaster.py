"""WebSocket trace event broadcaster.

Broadcasts Langfuse span events to connected WebSocket clients
in real-time.  Events are scoped by ``task_id`` so each client
only receives events for the task they're watching.

Architecture: src/api/ subscribes to this broadcaster to push
trace events to the dashboard via ``WS /events/stream``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


class TraceEvent:
    """A structured trace event to be broadcast over WebSocket."""

    __slots__ = (
        "type",
        "task_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "name",
        "span_type",
        "started_at",
        "ended_at",
        "tokens_in",
        "tokens_out",
        "cached_tokens",
        "cost_usd",
        "status",
        "metadata",
    )

    def __init__(
        self,
        *,
        type: str,
        task_id: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        name: str = "",
        span_type: str = "span",
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cached_tokens: int = 0,
        cost_usd: Decimal = Decimal("0"),
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.type = type
        self.task_id = task_id
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.span_type = span_type
        self.started_at = started_at or datetime.now(UTC)
        self.ended_at = ended_at
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cached_tokens = cached_tokens
        self.cost_usd = cost_usd
        self.status = status
        self.metadata = metadata or {}

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for WebSocket transmission."""
        data = {
            "type": self.type,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cached_tokens": self.cached_tokens,
            "cost_usd": str(self.cost_usd),
            "status": self.status,
            "metadata": self.metadata,
        }
        return json.dumps(data, default=str)


class TraceBroadcaster:
    """Manages WebSocket subscriptions and broadcasts trace events.

    Each subscriber is keyed by task_id.  When a trace event is
    published for a task, all subscribers for that task receive
    the event as a JSON message.
    """

    def __init__(self) -> None:
        # task_id -> set of asyncio.Queue
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, task_id: str) -> asyncio.Queue[str]:
        """Subscribe to trace events for a given task_id.

        Returns a queue that will receive JSON-encoded trace events.
        """
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            if task_id not in self._subscribers:
                self._subscribers[task_id] = set()
            self._subscribers[task_id].add(queue)
        logger.debug("Subscribed to trace events for task %s", task_id)
        return queue

    async def unsubscribe(self, task_id: str, queue: asyncio.Queue[str]) -> None:
        """Unsubscribe a queue from trace events for a given task_id."""
        async with self._lock:
            if task_id in self._subscribers:
                self._subscribers[task_id].discard(queue)
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]
        logger.debug("Unsubscribed from trace events for task %s", task_id)

    async def publish(self, event: TraceEvent) -> None:
        """Broadcast a trace event to all subscribers for the event's task_id."""
        async with self._lock:
            subscribers = self._subscribers.get(event.task_id, set()).copy()

        if not subscribers:
            return

        message = event.to_json()
        dead_queues: list[asyncio.Queue[str]] = []

        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(
                    "Trace event queue full for task %s, dropping event", event.task_id
                )
                dead_queues.append(queue)

        # Clean up full queues
        if dead_queues:
            async with self._lock:
                for q in dead_queues:
                    if event.task_id in self._subscribers:
                        self._subscribers[event.task_id].discard(q)

    @property
    def subscriber_count(self) -> dict[str, int]:
        """Return the number of subscribers per task_id."""
        return {tid: len(queues) for tid, queues in self._subscribers.items()}


# ── Module-level singleton ─────────────────────────────────────────

_broadcaster: TraceBroadcaster | None = None


def get_trace_broadcaster() -> TraceBroadcaster:
    """Return the module-level TraceBroadcaster singleton."""
    global _broadcaster  # noqa: PLW0603
    if _broadcaster is None:
        _broadcaster = TraceBroadcaster()
    return _broadcaster
