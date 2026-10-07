"""BrowseComp-Plus MCP search/get_document adapter and deterministic read-only replay."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Protocol
from uuid import uuid4

from .config import RetrievalConfig
from .evidence import normalize_query, stable_hash
from .storage import Store


class RetrievalError(RuntimeError):
    pass


def _connection_failed(exc: Exception) -> bool:
    from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
    from mcp.shared.session import CONNECTION_CLOSED
    return (getattr(getattr(exc, "error", None), "code", None) == CONNECTION_CLOSED
            or isinstance(exc, (ConnectionError, BrokenResourceError, ClosedResourceError, EndOfStream)))


def normalize_docid(value: Any) -> str:
    """Preserve opaque string IDs; only integer IDs have a string equivalent."""
    if type(value) is int:
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    raise RetrievalError("docid must be a nonblank string or an integer")


class Retriever(Protocol):
    async def search(self, query: str) -> list[dict]: ...
    async def get_document(self, docid: str) -> dict | None: ...
    async def close(self) -> None: ...


def parse_search(value: Any) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("result", value.get("results"))
    if not isinstance(value, list):
        raise RetrievalError("Search result has no result/results array")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise RetrievalError("Search item has no docid")
        docid = normalize_docid(item.get("docid"))
        if not isinstance(item.get("snippet"), str):
            raise RetrievalError("Search item has no snippet")
        result.append({"docid": docid, "snippet": item["snippet"],
                       "score": item.get("score"), "title": item.get("title", "")})
    return result


def parse_document(value: Any) -> dict | None:
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise RetrievalError("Document result has no docid/text")
    return {"docid": normalize_docid(value.get("docid")), "text": value["text"], "title": value.get("title", "")}


def mcp_payload(response: Any) -> Any:
    if getattr(response, "isError", False):
        message = "\n".join(getattr(c, "text", "") for c in response.content)
        raise RetrievalError(message or "MCP tool returned isError")
    structured = getattr(response, "structuredContent", None)
    if structured is not None:
        return structured
    text = "\n".join(getattr(c, "text", "") for c in response.content if getattr(c, "type", "") == "text")
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RetrievalError("MCP tool content is not valid JSON") from exc


@dataclass
class _McpRequest:
    name: str
    arguments: dict
    reply: asyncio.Future
    cleaned: asyncio.Future
    enqueued: float
    deadline: float
    request_id: str = field(default_factory=lambda: uuid4().hex)
    task: asyncio.Task | None = None
    started: float | None = None
    ended: float | None = None
    cancel_reason: str | None = None
    session_generation: int = 0
    timeout_observed: float | None = None
    cancel_requested: float | None = None


class McpRetriever:
    def __init__(self, config: RetrievalConfig, store: Store | None = None):
        self.config = config
        self.store = store
        self.session = None
        self.gate = asyncio.Semaphore(config.max_inflight)
        self.search_gate = asyncio.Semaphore(config.max_search_inflight)
        self.owner_task: asyncio.Task | None = None
        self.owner_queue: asyncio.Queue | None = None
        self.owner_lock = asyncio.Lock()
        self.closed = False
        self.requests: dict[str, _McpRequest] = {}
        self.owner_failed = False
        self.session_generation = 0

    def _finish_request(self, request: _McpRequest):
        """Run only after the call task has exited, including semaphore cleanup."""
        if request.cleaned.done():
            return
        request.ended = time.monotonic()
        outcome = request.cancel_reason or "success"
        response = None
        error = None
        error_type = None
        error_message = None
        if request.task is not None:
            try:
                response = request.task.result()
            except asyncio.CancelledError:
                outcome = request.cancel_reason or "session_failure"
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                code = getattr(getattr(exc, "error", None), "code", None)
                if not request.cancel_reason:
                    if code == 408 or isinstance(exc, TimeoutError):
                        outcome = "call_timeout"
                    elif _connection_failed(exc):
                        outcome = "session_failure"
                    else:
                        outcome = "tool_error"
                error = RetrievalError(f"{request.name}: {error_type}: {exc}")
        if getattr(response, "isError", False) and outcome == "success":
            outcome = "tool_error"
        if not request.reply.done():
            if error is not None:
                request.reply.set_exception(error)
            elif request.cancel_reason or outcome == "session_failure":
                message = (f"timed out after {self.config.timeout_seconds} seconds ({outcome})"
                           if outcome.endswith("_timeout") else outcome)
                request.reply.set_exception(RetrievalError(f"{request.name}: {message}"))
            else:
                request.reply.set_result(response)
        self.requests.pop(request.request_id, None)
        fields = dict(
            request_id=request.request_id, tool_name=request.name,
            enqueued_monotonic=request.enqueued, call_started_monotonic=request.started,
            ended_monotonic=request.ended,
            queue_wait_seconds=(request.started if request.started is not None else request.ended) - request.enqueued,
            call_seconds=None if request.started is None else request.ended - request.started,
            total_seconds=request.ended - request.enqueued, outcome=outcome,
            session_generation=request.session_generation,
            error_type=error_type, error_message=error_message,
            deadline_monotonic=request.deadline,
            timeout_observed_monotonic=request.timeout_observed,
            deadline_lateness_seconds=(None if request.timeout_observed is None else
                                       max(0, request.timeout_observed - request.deadline)),
            cancel_requested_monotonic=request.cancel_requested,
            cancellation_cleanup_seconds=(None if request.cancel_requested is None else
                                          request.ended - request.cancel_requested))
        try:
            if self.store is not None:
                self.store.event("mcp_request", **fields)
            else:
                logging.getLogger(__name__).info("mcp_request %s", json.dumps(fields))
        finally:
            request.cleaned.set_result(None)

    async def open(self):
        await self._ensure_owner()
        return self

    async def _run_owner(self, queue: asyncio.Queue, ready: asyncio.Future):
        """Own the AnyIO transport scopes for their complete lifetime."""
        from mcp import StdioServerParameters
        from .mcp_session import ResilientClientSession
        import anyio
        calls: set[asyncio.Task] = set()
        pending: dict[str, _McpRequest] = {}
        stop_reason = "session_failure"

        async def invoke(session, request):
            async with AsyncExitStack() as admission:
                # Queued searches must not occupy the capacity reserved for fetches.
                if request.name == "search":
                    await admission.enter_async_context(self.search_gate)
                await admission.enter_async_context(self.gate)
                # The deadline may win before the owner receives the cancel message.
                if request.cancel_reason or request.reply.cancelled():
                    return
                if time.monotonic() >= request.deadline:
                    request.cancel_reason = "queue_timeout"
                    return
                request.started = time.monotonic()
                try:
                    return await session.call_tool(request.name, arguments=request.arguments)
                except Exception as exc:
                    if _connection_failed(exc) and not self.owner_failed:
                        # A receive loop may exit without ending the outer owner.
                        # Retire that generation in its owner, never in this task.
                        self.owner_failed = True
                        queue.put_nowait(("session_failed", None))
                    raise

        def completed(request, task):
            calls.discard(task)
            pending.pop(request.request_id, None)
            self._finish_request(request)

        try:
            async with AsyncExitStack() as stack:
                if self.config.transport == "http":
                    from mcp.client.streamable_http import streamablehttp_client
                    transport = streamablehttp_client(
                        self.config.url, timeout=self.config.timeout_seconds,
                        sse_read_timeout=self.config.timeout_seconds)
                else:
                    from mcp.client.stdio import stdio_client
                    transport = stdio_client(StdioServerParameters(
                        command=self.config.command, args=self.config.args, cwd=self.config.cwd,
                        env={**os.environ, **self.config.env}))
                streams = await stack.enter_async_context(transport)
                session = await stack.enter_async_context(ResilientClientSession(
                    streams[0], streams[1], read_timeout_seconds=None))
                await session.initialize()
                tools = await session.list_tools()
                missing = {"search", "get_document"} - {tool.name for tool in tools.tools}
                if missing:
                    raise RetrievalError(f"MCP is missing required tools: {sorted(missing)}")
                self.session = session
                if not ready.done():
                    ready.set_result(None)
                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            stop_reason = "closed"
                            break
                        action, request = item
                        if action == "session_failed":
                            break
                        if action == "cancel":
                            task = request.task
                            if task is not None and not task.done() and not task.cancelling():
                                task.cancel()
                        elif request.cancel_reason or request.reply.cancelled():
                            self._finish_request(request)
                        else:
                            pending[request.request_id] = request
                            task = asyncio.create_task(invoke(session, request))
                            request.task = task
                            calls.add(task)
                            task.add_done_callback(lambda task, request=request: completed(request, task))
                finally:
                    for request in pending.values():
                        request.cancel_reason = request.cancel_reason or stop_reason
                    for task in tuple(calls):
                        if not task.cancelling():
                            task.cancel()
                    # Transport failure can cancel the owner's AnyIO scope. Keep
                    # cleanup in this owner, before it exits the shared session.
                    with anyio.CancelScope(shield=True):
                        if calls:
                            await asyncio.gather(*tuple(calls), return_exceptions=True)
        except BaseException as exc:
            error = exc if isinstance(exc, RetrievalError) else RetrievalError(
                f"MCP session failed: {type(exc).__name__}: {exc}")
            if not ready.done():
                ready.set_exception(error)
        finally:
            while not queue.empty():
                item = queue.get_nowait()
                if item is not None:
                    _, request = item
                    if request is None:
                        continue
                    request.cancel_reason = request.cancel_reason or stop_reason
                    self._finish_request(request)
            self.session = None

    async def _ensure_owner(self):
        async with self.owner_lock:
            if self.closed:
                raise RetrievalError("MCP retriever is closed")
            if self.owner_failed and self.owner_task is not None:
                await asyncio.shield(self.owner_task)
                if self.closed:
                    raise RetrievalError("MCP retriever is closed")
            if self.owner_task is not None and not self.owner_task.done():
                return
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            ready = loop.create_future()
            self.owner_queue = queue
            self.owner_failed = False
            self.session_generation += 1
            self.owner_task = asyncio.create_task(self._run_owner(queue, ready))
            # Retrieving the exception in the callback avoids an unhandled-task
            # warning; the next call still sees done() and creates a new owner.
            self.owner_task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception())
            await ready

    async def _call(self, name: str, arguments: dict):
        await self._ensure_owner()
        owner = self.owner_task
        # A transport may fail after signaling ready, before this caller resumes.
        if owner.done():
            raise RetrievalError(f"{name}: MCP session stopped")
        loop = asyncio.get_running_loop()
        reply = loop.create_future()
        enqueued = time.monotonic()
        request = _McpRequest(name, arguments, reply, loop.create_future(), enqueued,
                              enqueued + self.config.timeout_seconds,
                              session_generation=self.session_generation)
        queue = self.owner_queue
        self.requests[request.request_id] = request
        queue.put_nowait(("call", request))
        try:
            done, _ = await asyncio.wait(
                {reply, owner}, timeout=self.config.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED)
            if reply in done:
                response = reply.result()
            elif owner in done:
                raise RetrievalError(f"{name}: MCP session stopped")
            else:
                request.timeout_observed = time.monotonic()
                request.cancel_reason = "queue_timeout" if request.started is None else "call_timeout"
                raise RetrievalError(
                    f"{name}: timed out after {self.config.timeout_seconds} seconds ({request.cancel_reason})")
            return mcp_payload(response)
        except asyncio.CancelledError:
            request.cancel_reason = "caller_cancelled"
            raise
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"{name}: {exc}") from exc
        finally:
            if not reply.done():
                reply.cancel()
            elif not reply.cancelled():
                reply.exception()
            if not request.cleaned.done():
                # Never cancel the owner or close its AnyIO scopes from a caller.
                # The owner signals completion only after the per-call task exits.
                request.cancel_requested = time.monotonic()
                queue.put_nowait(("cancel", request))
                await asyncio.shield(request.cleaned)

    async def search(self, query: str) -> list[dict]:
        return parse_search(await self._call("search", {"query": query}))[:self.config.top_k]

    async def get_document(self, docid: str) -> dict | None:
        docid = normalize_docid(docid)
        document = parse_document(await self._call("get_document", {"docid": docid}))
        if document is not None and document["docid"] != docid:
            raise RetrievalError("get_document returned a different docid")
        return document

    async def close(self):
        self.closed = True
        if self.owner_task is None:
            return
        if not self.owner_task.done():
            await self.owner_queue.put(None)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.owner_task), timeout=self.config.timeout_seconds)
            except (asyncio.TimeoutError, TimeoutError):
                self.owner_task.cancel()
                await asyncio.gather(self.owner_task, return_exceptions=True)


class FixtureRetriever:
    """Small synthetic corpus for tests/demo, never used silently as production search."""
    def __init__(self, data: dict):
        self.data = deepcopy(data)
        self.calls = []
        self.queries = {normalize_query(k): v for k, v in data.get("search", {}).items()}

    async def search(self, query: str) -> list[dict]:
        self.calls.append(("search", query))
        ids = self.queries.get(normalize_query(query), [])
        return [{"docid": str(i), "snippet": self.data["documents"][str(i)]["text"][:500], "score": 1.0}
                for i in ids]

    async def get_document(self, docid: str) -> dict | None:
        self.calls.append(("get_document", docid))
        value = self.data["documents"].get(docid)
        return {"docid": docid, **deepcopy(value)} if value else None

    async def close(self):
        pass


class RecordedRetriever:
    """Persist every logical tool result; memory caching is bounded and shared across cells.

    Logical search/fetch budgets count calls even on cache hits. Telemetry separately
    records physical backend requests. Corpus text is retained only in the journal.
    """
    def __init__(self, inner: Retriever, store: Store, capacity: int = 2048):
        self.inner, self.store, self.capacity = inner, store, capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.locks: dict[str, asyncio.Lock] = {}

    async def _call(self, tool: str, argument: str, operation_id: str):
        identity = stable_hash([tool, argument])
        key = "tool:" + operation_id
        previous = self.store.get(key, identity)
        if previous:
            if previous["status"] == "success":
                return deepcopy(previous["value"])
            if previous["status"] == "rejected":
                raise RetrievalError(previous["value"]["error"])
        lock = self.locks.setdefault(identity, asyncio.Lock())
        async with lock:
            started = time.monotonic()
            cache_hit = identity in self.cache
            try:
                if cache_hit:
                    value = self.cache.pop(identity)
                    self.cache[identity] = value
                elif tool == "search":
                    value = await self.inner.search(argument)
                else:
                    value = await self.inner.get_document(argument)
                if tool == "get_document":
                    value = parse_document(value)
                    if value is not None and value["docid"] != argument:
                        raise RetrievalError("get_document returned a different docid")
                if self.capacity:
                    self.cache[identity] = deepcopy(value)
                    while len(self.cache) > self.capacity:
                        self.cache.popitem(last=False)
                self.store.put(key, identity, "success", value)
                self.store.event("tool_call", operation_id=operation_id, tool=tool,
                                 argument=argument, success=True, cache_hit=cache_hit,
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 docids=[x["docid"] for x in value] if tool == "search" else [argument])
                return deepcopy(value)
            except RetrievalError as exc:
                self.store.put(key, identity, "rejected", {"error": str(exc)})
                self.store.event("tool_call", operation_id=operation_id, tool=tool,
                                 argument=argument, success=False, cache_hit=cache_hit,
                                 duration_ms=(time.monotonic() - started) * 1000, error=str(exc))
                raise

    async def search(self, query: str, operation_id: str) -> list[dict]:
        return await self._call("search", query, operation_id)

    async def get_document(self, docid: str, operation_id: str) -> dict | None:
        return await self._call("get_document", normalize_docid(docid), operation_id)

    async def close(self):
        await self.inner.close()
