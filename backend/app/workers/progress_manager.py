"""Thread-safe SSE progress hub with bounded history and slow-client isolation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import threading
from typing import Any

from backend.app.utils.serialization import sanitize_for_json, utc_now


TERMINAL_SSE_EVENTS = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class StreamEvent:
    event_id: int
    event_type: str
    data: dict[str, Any]

    def format_sse(self) -> str:
        payload = json.dumps(sanitize_for_json(self.data), ensure_ascii=False, allow_nan=False)
        return f"id: {self.event_id}\nevent: {self.event_type}\ndata: {payload}\n\n"


class _Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int = 100) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=maxsize)

    def push_threadsafe(self, event: StreamEvent | None) -> None:
        if self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue_nowait, event)
        except RuntimeError:
            pass

    def _enqueue_nowait(self, event: StreamEvent | None) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


class ProgressManager:
    """Manages real-time progress events, bounded replay buffers, and SSE clients."""

    def __init__(self, history_limit: int = 200) -> None:
        self._history_limit = history_limit
        self._lock = threading.RLock()
        self._counters: dict[str, int] = {}
        self._history: dict[str, deque[StreamEvent]] = {}
        self._subscribers: dict[str, set[_Subscriber]] = {}
        self._shutdown = False

    def publish(self, job_id: str, event_type: str, data: dict[str, Any]) -> StreamEvent:
        with self._lock:
            seq = self._counters.get(job_id, 0) + 1
            self._counters[job_id] = seq
            payload = {
                "job_id": job_id,
                "event_type": event_type,
                "timestamp": utc_now().isoformat(),
                **sanitize_for_json(data),
            }
            event = StreamEvent(event_id=seq, event_type=event_type, data=payload)
            buf = self._history.setdefault(job_id, deque(maxlen=self._history_limit))
            buf.append(event)
            subs = list(self._subscribers.get(job_id, ()))

        for sub in subs:
            sub.push_threadsafe(event)
        return event

    def subscribe(
        self,
        job_id: str,
        loop: asyncio.AbstractEventLoop,
        *,
        last_event_id: int | None = None,
    ) -> tuple[_Subscriber, list[StreamEvent]]:
        sub = _Subscriber(loop=loop)
        with self._lock:
            history = list(self._history.get(job_id, ()))
            if last_event_id is not None:
                replay = [item for item in history if item.event_id > last_event_id]
            else:
                replay = []
            if not self._shutdown:
                self._subscribers.setdefault(job_id, set()).add(sub)
            else:
                sub.push_threadsafe(None)
        return sub, replay

    def unsubscribe(self, job_id: str, sub: _Subscriber) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(sub)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            all_subs = [
                sub
                for subs in self._subscribers.values()
                for sub in subs
            ]
            self._subscribers.clear()
        for sub in all_subs:
            sub.push_threadsafe(None)
