import pytest
from bcgraph.storage import Store

@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path / 'run')
    yield instance
    instance.close()

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

@pytest.fixture
def fake_mcp(monkeypatch):
    """Exercise the real connection owner, replacing only the SDK boundary."""
    pytest.importorskip('mcp')

    class Session:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.cleanup = asyncio.Event()
            self.cleanup.set()
            self.calls = []
            self.active = 0
            self.closed = False
            self.fail_transport = asyncio.Event()

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in ('search', 'get_document')])

        async def call_tool(self, name, arguments):
            query = arguments.get('query', '')
            self.calls.append(query)
            self.active += 1
            try:
                if query.startswith('slow'):
                    self.entered.set()
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        await self.cleanup.wait()
                        raise
                return SimpleNamespace(isError=False, structuredContent={'result': []}, content=[])
            finally:
                self.active -= 1

    session = Session()

    @asynccontextmanager
    async def transport(*args, **kwargs):
        import anyio
        owner = asyncio.current_task()
        async def fail():
            await session.fail_transport.wait()
            raise ConnectionError('fake transport failure')
        async with anyio.create_task_group() as group:
            group.start_soon(fail)
            try:
                yield (None, None, None)
            finally:
                assert asyncio.current_task() is owner
                group.cancel_scope.cancel()

    @asynccontextmanager
    async def client_session(*args, **kwargs):
        owner = asyncio.current_task()
        try:
            yield session
        finally:
            assert asyncio.current_task() is owner
            session.closed = True

    monkeypatch.setattr('bcgraph.mcp_session.ResilientClientSession', client_session)
    monkeypatch.setattr('mcp.client.streamable_http.streamablehttp_client', transport)
    return session
