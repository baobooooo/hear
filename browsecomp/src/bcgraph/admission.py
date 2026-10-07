"""Process-wide inference admission with continuation priority and bounded starvation."""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import itertools
import time
from typing import AsyncIterator


@dataclass
class _Waiter:
    priority: int
    sequence: int
    created: float
    future: asyncio.Future
    granted: bool = False


class PriorityGate:
    def __init__(self, capacity: int, aging_seconds: float = 10):
        if capacity < 1 or aging_seconds <= 0:
            raise ValueError("Invalid gate settings")
        self.capacity, self.aging_seconds = capacity, aging_seconds
        self.active = self.peak_active = 0
        self._sequence = itertools.count()
        self._queue: list[_Waiter] = []

    def _drain(self):
        while self.active < self.capacity and self._queue:
            now = time.monotonic()
            w = min(self._queue, key=lambda x: (
                x.priority - (now - x.created) / self.aging_seconds, x.sequence))
            self._queue.remove(w)
            if w.future.cancelled():
                continue
            w.granted = True
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            w.future.set_result(None)

    @asynccontextmanager
    async def slot(self, continuation: bool = False) -> AsyncIterator[float]:
        created = time.monotonic()
        w = _Waiter(0 if continuation else 1, next(self._sequence), created,
                    asyncio.get_running_loop().create_future())
        self._queue.append(w)
        self._drain()
        try:
            await asyncio.shield(w.future)
            yield (time.monotonic() - created) * 1000
        finally:
            # Handles cancellation both before and after a permit was granted.
            if w.granted:
                self.active -= 1
            elif w in self._queue:
                self._queue.remove(w)
                w.future.cancel()
            self._drain()

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "active": self.active,
                "pending": len(self._queue), "peak_active": self.peak_active}
