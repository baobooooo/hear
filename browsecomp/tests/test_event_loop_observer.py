import asyncio
from types import SimpleNamespace

import pytest

from bcgraph import app


@pytest.mark.asyncio
async def test_loop_observer_records_lateness_and_propagates_cancellation(monkeypatch):
    times = iter([10.0, 11.25, 11.25])
    monkeypatch.setattr(app, "time", SimpleNamespace(monotonic=lambda: next(times)))
    samples = []
    waits = []
    async def sleep(interval):
        waits.append(interval)
        if len(waits) == 2:
            raise asyncio.CancelledError
    monkeypatch.setattr(app, "asyncio", SimpleNamespace(sleep=sleep))
    store = SimpleNamespace(event=lambda kind, **fields: samples.append({"kind": kind, **fields}))
    with pytest.raises(asyncio.CancelledError):
        await app.observe_event_loop(store)
    assert samples == [{"kind": "event_loop_lag", "lag_ms": 250.0, "interval_seconds": 1.0}]
