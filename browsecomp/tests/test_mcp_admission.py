"""Exercise admission through the real MCP owner with a controllable SDK boundary."""
import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

from bcgraph.config import RetrievalConfig
from bcgraph.retrieval import McpRetriever, RetrievalError


@pytest.mark.asyncio
async def test_search_backlog_leaves_two_fetch_slots_and_total_stays_eight(fake_mcp, monkeypatch):
    entered = asyncio.Queue()
    release = asyncio.Event()
    active, peak = Counter(), Counter()

    async def call(name, arguments):
        active[name] += 1
        peak[name] = max(peak[name], active[name])
        peak['total'] = max(peak['total'], sum(active.values()))
        entered.put_nowait(name)
        try:
            await release.wait()
            return SimpleNamespace(isError=False, structuredContent={'result': []}, content=[])
        finally:
            active[name] -= 1

    monkeypatch.setattr(fake_mcp, 'call_tool', call)
    client = await McpRetriever(RetrievalConfig(timeout_seconds=5)).open()
    tasks = [asyncio.create_task(client._call('search', {'query': str(i)})) for i in range(12)]
    try:
        for _ in range(6):
            assert await asyncio.wait_for(entered.get(), 1) == 'search'
        tasks += [asyncio.create_task(client._call('get_document', {'docid': str(i)})) for i in range(3)]
        for _ in range(2):
            assert await asyncio.wait_for(entered.get(), 1) == 'get_document'
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(entered.get(), .03)
        assert active == {'search': 6, 'get_document': 2}
        release.set()
        await asyncio.gather(*tasks)
        assert peak['search'] == 6
        assert peak['total'] == 8
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_documents_can_use_all_eight_slots(fake_mcp, monkeypatch):
    entered = asyncio.Queue()
    release = asyncio.Event()

    async def call(name, arguments):
        entered.put_nowait(name)
        await release.wait()
        return SimpleNamespace(isError=False, structuredContent={'result': []}, content=[])

    monkeypatch.setattr(fake_mcp, 'call_tool', call)
    client = await McpRetriever(RetrievalConfig(timeout_seconds=5)).open()
    tasks = [asyncio.create_task(client._call('get_document', {'docid': str(i)})) for i in range(9)]
    try:
        for _ in range(8):
            assert await asyncio.wait_for(entered.get(), 1) == 'get_document'
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(entered.get(), .03)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_gate', ['search_gate', 'gate'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_waiting_timeout_or_cancel_releases_admission(fake_mcp, blocked_gate, cancel):
    client = await McpRetriever(RetrievalConfig(
        max_inflight=1, max_search_inflight=1, timeout_seconds=.1)).open()
    gate = getattr(client, blocked_gate)
    await gate.acquire()
    task = asyncio.create_task(client.search('queued'))
    try:
        if cancel:
            await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RetrievalError, match='queue_timeout'):
                await task
        assert fake_mcp.calls == []
        gate.release()
        assert await asyncio.wait_for(client.search('next'), 1) == []
        assert not client.requests
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_running_cancel_holds_both_slots_until_cleanup(fake_mcp):
    client = await McpRetriever(RetrievalConfig(
        max_inflight=1, max_search_inflight=1, timeout_seconds=1)).open()
    fake_mcp.cleanup.clear()
    task = asyncio.create_task(client.search('slow'))
    try:
        await asyncio.wait_for(fake_mcp.entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(fake_mcp.cancelled.wait(), 1)
        assert client.search_gate.locked() and client.gate.locked()
        fake_mcp.cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await client.search('next') == []
        assert not client.search_gate.locked() and not client.gate.locked()
    finally:
        fake_mcp.cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()


def test_search_limit_must_be_positive():
    with pytest.raises(ValueError):
        RetrievalConfig(max_search_inflight=0)
