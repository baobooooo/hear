"""Regressions for response/cancel races and a live owner with a dead session."""
import asyncio
import json

import anyio
import pytest
from anyio.streams.memory import MemoryObjectSendStream
from mcp import types
from mcp.shared.message import SessionMessage

from bcgraph.config import RetrievalConfig
from bcgraph.retrieval import McpRetriever, RetrievalError
from bcgraph.mcp_session import ResilientClientSession as ClientSession


@pytest.mark.asyncio
@pytest.mark.parametrize('error_response', [False, True])
async def test_cancel_during_response_delivery_preserves_other_requests(monkeypatch, error_response):
    incoming, read = anyio.create_memory_object_stream(8)
    write, outgoing = anyio.create_memory_object_stream(8)
    entered, release = asyncio.Event(), asyncio.Event()
    target = None
    original_send = MemoryObjectSendStream.send

    async def delayed_send(stream, message):
        if stream is target:
            entered.set()
            await release.wait()
        return await original_send(stream, message)

    async def respond(request, error=False):
        response = (types.JSONRPCError(jsonrpc='2.0', id=request.message.root.id,
                                      error=types.ErrorData(code=-32603, message='late error'))
                    if error else types.JSONRPCResponse(
                        jsonrpc='2.0', id=request.message.root.id, result={'content': []}))
        await incoming.send(SessionMessage(types.JSONRPCMessage(response)))

    monkeypatch.setattr(MemoryObjectSendStream, 'send', delayed_send)
    async with incoming, outgoing, ClientSession(read, write) as session:
        listing = asyncio.create_task(session.list_tools())
        listing_request = await outgoing.receive()
        await incoming.send(SessionMessage(types.JSONRPCMessage(types.JSONRPCResponse(
            jsonrpc='2.0', id=listing_request.message.root.id,
            result={'tools': [{'name': 'search', 'inputSchema': {'type': 'object'}}]}))))
        await listing
        first = asyncio.create_task(session.call_tool('search', {'query': 'cancel'}))
        request = await outgoing.receive()
        target = session._response_streams[request.message.root.id]
        second = asyncio.create_task(session.call_tool('search', {'query': 'keep'}))
        second_request = await outgoing.receive()
        await respond(request, error_response)
        await asyncio.wait_for(entered.wait(), 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        await respond(second_request)
        assert not (await asyncio.wait_for(second, 1)).isError
        third = asyncio.create_task(session.call_tool('search', {'query': 'next'}))
        await respond(await outgoing.receive())
        assert not (await asyncio.wait_for(third, 1)).isError
        assert not session._response_streams


@pytest.mark.asyncio
async def test_dead_session_reconnects_without_replaying_failed_call(fake_mcp, monkeypatch, store):
    original_initialize = fake_mcp.initialize
    original_call = fake_mcp.call_tool
    generations = 0

    async def initialize():
        nonlocal generations
        generations += 1
        await original_initialize()

    async def call(name, arguments):
        if generations == 1:
            raise anyio.ClosedResourceError()
        return await original_call(name, arguments)

    monkeypatch.setattr(fake_mcp, 'initialize', initialize)
    monkeypatch.setattr(fake_mcp, 'call_tool', call)
    client = await McpRetriever(RetrievalConfig(timeout_seconds=1), store=store).open()
    try:
        with pytest.raises(RetrievalError):
            await client.search('broken')
        assert await asyncio.gather(*(client.search('next') for _ in range(4))) == [[], [], [], []]
        assert generations == 2
        assert fake_mcp.calls == ['next'] * 4
        assert not client.requests
        events = [json.loads(line) for line in (store.meta/'events.jsonl').read_text().splitlines()]
        assert events[0]['error_type'] == 'ClosedResourceError'
        assert events[0]['session_generation'] < events[1]['session_generation']
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_timeout_telemetry_separates_deadline_and_cleanup(fake_mcp, store):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.02), store=store).open()
    fake_mcp.cleanup.clear()
    call = asyncio.create_task(client.search('slow'))
    try:
        await asyncio.wait_for(fake_mcp.cancelled.wait(), 1)
        await asyncio.sleep(.03)
        fake_mcp.cleanup.set()
        with pytest.raises(RetrievalError, match='call_timeout'):
            await call
        event = json.loads((store.meta/'events.jsonl').read_text().splitlines()[-1])
        assert event['cancellation_cleanup_seconds'] >= .03
        assert event['deadline_lateness_seconds'] >= 0
        assert event['deadline_monotonic'] <= event['timeout_observed_monotonic']
        assert event['timeout_observed_monotonic'] <= event['cancel_requested_monotonic']
        assert event['cancel_requested_monotonic'] <= event['ended_monotonic']
    finally:
        fake_mcp.cleanup.set()
        await asyncio.gather(call, return_exceptions=True)
        await client.close()
