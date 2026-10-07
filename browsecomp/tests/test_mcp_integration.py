"""Real MCP stdio subprocess test; no external search service required."""
import sys
from pathlib import Path
import pytest
pytest.importorskip('mcp',reason='MCP SDK required for protocol integration')
from bcgraph.config import RetrievalConfig
from bcgraph.retrieval import McpRetriever, RetrievalError
import asyncio
import os
import socket
import subprocess
import json


@pytest.mark.asyncio
async def test_queued_timeout_never_calls_tool(fake_mcp):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.03, max_inflight=1)).open()
    await client.gate.acquire()
    try:
        with pytest.raises(RetrievalError, match='timed out'):
            await client.search('queued')
        client.gate.release()
        await asyncio.sleep(.02)
        assert fake_mcp.calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_running_timeout_cancels_call(fake_mcp):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.03, max_inflight=1)).open()
    try:
        with pytest.raises(RetrievalError, match='timed out'):
            await client.search('slow')
        assert fake_mcp.cancelled.is_set()
        assert fake_mcp.active == 0
        assert await client.search('fast') == []
        assert not fake_mcp.closed
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancellation_keeps_slot_until_cleanup_and_does_not_block_owner(fake_mcp):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.03, max_inflight=1)).open()
    fake_mcp.cleanup.clear()
    slow = asyncio.create_task(client.search('slow'))
    try:
        await asyncio.wait_for(fake_mcp.cancelled.wait(), 1)
        assert client.gate.locked()
        assert fake_mcp.active == 1
        assert not slow.done()
        client.config.timeout_seconds = 1
        waiting = asyncio.create_task(client.search('never_sent'))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, .2)
        assert fake_mcp.calls == ['slow']
        assert not client.owner_task.done()
        fake_mcp.cleanup.set()
        with pytest.raises(RetrievalError, match='call_timeout'):
            await slow
        assert await client.search('fast') == []
        assert not client.requests
    finally:
        fake_mcp.cleanup.set()
        await asyncio.gather(slow, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_repeated_timeouts_leave_no_registry_or_capacity_backlog(fake_mcp):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.01, max_inflight=2)).open()
    try:
        for _ in range(5):
            replies = await asyncio.gather(*(client.search('slow') for _ in range(6)), return_exceptions=True)
            assert all(isinstance(r, RetrievalError) for r in replies)
            assert not client.requests
            assert fake_mcp.active == 0
            await asyncio.wait_for(client.gate.acquire(), .1)
            await asyncio.wait_for(client.gate.acquire(), .1)
            assert client.gate.locked()
            client.gate.release()
            client.gate.release()
        assert await client.search('fast') == []
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('delay', [0, .008, .01, .012])
async def test_timeout_return_race_has_one_completion(fake_mcp, store, delay):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.01), store=store).open()
    loop = asyncio.get_running_loop()
    errors = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    try:
        for _ in range(10):
            fake_mcp.release.clear()
            timer = loop.call_later(delay, fake_mcp.release.set)
            try:
                await client.search('slow')
            except RetrievalError as exc:
                assert 'timed out' in str(exc)
            finally:
                timer.cancel()
            assert not client.requests
            assert fake_mcp.active == 0
        await asyncio.sleep(0)
        events = [json.loads(line) for line in (store.meta/'events.jsonl').read_text().splitlines()]
        assert len(events) == len({e['request_id'] for e in events}) == 10
        assert not errors
    finally:
        await client.close()
        loop.set_exception_handler(old_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True, 'transport'])
async def test_owner_shutdown_finishes_active_and_queued_requests(fake_mcp, store, failure):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=1, max_inflight=1), store=store).open()
    calls = [asyncio.create_task(client.search('slow')) for _ in range(4)]
    await fake_mcp.entered.wait()
    if failure == 'transport':
        fake_mcp.fail_transport.set()
    elif failure:
        client.owner_task.cancel()
    else:
        await client.close()
    results = await asyncio.wait_for(asyncio.gather(*calls, return_exceptions=True), 1)
    await client.close()
    assert all(isinstance(value, RetrievalError) for value in results)
    assert not client.requests
    assert fake_mcp.active == 0
    assert fake_mcp.closed
    events = [json.loads(line) for line in (store.meta/'events.jsonl').read_text().splitlines()]
    assert len(events) == 4
    assert {e['outcome'] for e in events} == {'session_failure' if failure else 'closed'}


@pytest.mark.asyncio
async def test_cancelling_one_active_request_keeps_other_active_request(fake_mcp):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=1, max_inflight=2)).open()
    first = asyncio.create_task(client.search('slow-first'))
    second = asyncio.create_task(client.search('slow-second'))
    try:
        async with asyncio.timeout(1):
            while fake_mcp.active != 2:
                await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()
        assert fake_mcp.active == 1
        assert not fake_mcp.closed
        fake_mcp.release.set()
        assert await second == []
        assert await client.search('fast') == []
        assert not client.requests
    finally:
        fake_mcp.release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_phase_timings_and_error_classification(fake_mcp, store, monkeypatch):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.02, max_inflight=1), store=store).open()
    try:
        await client.gate.acquire()
        with pytest.raises(RetrievalError, match='queue_timeout'):
            await client.search('queued')
        client.gate.release()
        with pytest.raises(RetrievalError, match='call_timeout'):
            await client.search('slow')
        fake_mcp.entered.clear()
        call = asyncio.create_task(client.search('slow-cancel'))
        await fake_mcp.entered.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        async def failed(*args, **kwargs):
            raise ConnectionError('session unavailable')
        monkeypatch.setattr(fake_mcp, 'call_tool', failed)
        with pytest.raises(RetrievalError, match='session unavailable'):
            await client.search('failed')
        events = [json.loads(line) for line in (store.meta/'events.jsonl').read_text().splitlines()]
        assert [e['outcome'] for e in events] == [
            'queue_timeout', 'call_timeout', 'caller_cancelled', 'session_failure']
        assert events[0]['call_seconds'] is None
        for event in events:
            assert event['tool_name'] == 'search'
            assert event['queue_wait_seconds'] >= 0
            assert event['total_seconds'] == pytest.approx(
                event['queue_wait_seconds'] + (event['call_seconds'] or 0))
        assert all(e['call_seconds'] is not None for e in events[1:])
    finally:
        await client.close()




@pytest.mark.asyncio
async def test_owner_stopped_at_admission_finishes_request(fake_mcp, monkeypatch):
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.02)).open()
    client.owner_task.cancel()
    await client.owner_task
    async def already_ready():
        pass
    monkeypatch.setattr(client, '_ensure_owner', already_ready)
    with pytest.raises(RetrievalError, match='session stopped'):
        await asyncio.wait_for(client.search('never_sent'), .2)
    assert not client.requests
    assert fake_mcp.calls == []
    await client.close()


@pytest.mark.mcp
@pytest.mark.asyncio
async def test_actual_stdio_mcp_lifecycle():
    config=RetrievalConfig(transport='stdio',command=sys.executable,
                           args=[str(Path(__file__).parent/'fixtures'/'mcp_server.py')])
    client=await McpRetriever(config).open()
    try:
        found=await client.search('test')
        assert found[0]['docid']=='d1'
        assert (await client.get_document('d1'))['text']=='Synthetic document text.'
    finally: await client.close()


@pytest.mark.mcp
@pytest.mark.asyncio
async def test_streamable_http_mcp_shared_session_and_shutdown():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([sys.executable, str(Path(__file__).parent/'fixtures'/'mcp_server.py')],
                              env={**os.environ, 'TEST_MCP_PORT': str(port)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = None
    try:
        for _ in range(100):
            if server.poll() is not None:
                pytest.fail('Test MCP server exited during startup')
            try:
                _, writer = await asyncio.open_connection('127.0.0.1', port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            pytest.fail('Test MCP HTTP startup timed out')
        client = await McpRetriever(RetrievalConfig(url=f'http://127.0.0.1:{port}/mcp/')).open()
        replies = await asyncio.gather(*(client.search('test') for _ in range(4)))
        assert all(row[0]['docid'] == 'd1' for row in replies)
        assert (await client.get_document('d1'))['text'] == 'Synthetic document text.'
    finally:
        if client is not None:
            await client.close()
        server.terminate()
        await asyncio.to_thread(server.wait, timeout=10)


@pytest.mark.mcp
@pytest.mark.asyncio
async def test_streamable_http_server_restart_isolated_from_caller():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [sys.executable, str(Path(__file__).parent/'fixtures'/'mcp_server.py')]
    env = {**os.environ, 'TEST_MCP_PORT': str(port)}

    async def start_server():
        process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail('Test MCP server exited during startup')
            try:
                _, writer = await asyncio.open_connection('127.0.0.1', port)
                writer.close(); await writer.wait_closed()
                return process
            except OSError:
                await asyncio.sleep(0.05)
        pytest.fail('Test MCP HTTP startup timed out')

    server = await start_server()
    client = await McpRetriever(RetrievalConfig(
        url=f'http://127.0.0.1:{port}/mcp/', timeout_seconds=1)).open()
    try:
        assert (await client.search('test'))[0]['docid'] == 'd1'
        server.terminate(); await asyncio.to_thread(server.wait, timeout=10)
        with pytest.raises(RetrievalError):
            await client.search('during restart')
        assert not asyncio.current_task().cancelling()
        server = await start_server()
        assert (await client.search('after restart'))[0]['docid'] == 'd1'
    finally:
        await client.close()
        if server.poll() is None:
            server.terminate(); await asyncio.to_thread(server.wait, timeout=10)
