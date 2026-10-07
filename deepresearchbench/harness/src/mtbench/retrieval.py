from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
import fcntl
import hashlib
import itertools
import multiprocessing
import os
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
import pypdfium2 as pdfium
from trafilatura import baseline, extract as extract_main_text
from trafilatura import html2txt

from .config import RetrievalConfig
from .models import Document
from .trajectory import TraceWriter


_MIN_EXTRACTED_TEXT_CHARS = 300
_MAX_HTML_BYTES = 5_000_000
_MAX_PDF_BYTES = 25_000_000

# PDF extraction is disabled: libpdfium is not thread-safe and crashed a runner
# (SIGSEGV in libpdfium.so, 2026-09-19).  PDF results are skipped, not fetched.
_PDF_EXTRACTION_ENABLED = False


def _is_pdf_url(url: str) -> bool:
    return url.lower().split("?", 1)[0].split("#", 1)[0].endswith(".pdf")


def _is_pdf_response(url: str, content_type: str) -> bool:
    return "application/pdf" in content_type or _is_pdf_url(url)


_EXTRACTION_TIMEOUT_SECONDS = int(os.environ.get("MTBENCH_EXTRACTION_TIMEOUT", "120"))
_EXTRACTOR_WORKERS = max(1, int(os.environ.get("MTBENCH_EXTRACTOR_WORKERS", "4")))
_FETCH_BATCH_SIZE = max(1, int(os.environ.get("MTBENCH_FETCH_BATCH", "5")))
_SEARCH_PAGE_BATCH = max(1, int(os.environ.get("MTBENCH_SEARCH_PAGE_BATCH", "3")))
_extractor_pool: concurrent.futures.ProcessPoolExecutor | None = None
_extractor_pool_lock = threading.Lock()


def _extractor_executor() -> concurrent.futures.ProcessPoolExecutor:
    """One forkserver-backed worker so a native crash cannot kill the workflow."""
    global _extractor_pool
    with _extractor_pool_lock:
        if _extractor_pool is None:
            _extractor_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=_EXTRACTOR_WORKERS,
                mp_context=multiprocessing.get_context("forkserver"),
            )
        return _extractor_pool


def _reset_extractor_executor(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    global _extractor_pool
    with _extractor_pool_lock:
        if _extractor_pool is executor:
            _extractor_pool = None
    executor.shutdown(wait=False, cancel_futures=True)


async def _run_isolated(func, *args):
    """Run a native-library extractor out of process; crashes become ValueError."""
    loop = asyncio.get_running_loop()
    executor = _extractor_executor()
    try:
        future = loop.run_in_executor(executor, func, *args)
        return await asyncio.wait_for(future, timeout=_EXTRACTION_TIMEOUT_SECONDS)
    except concurrent.futures.process.BrokenProcessPool as exc:
        _reset_extractor_executor(executor)
        raise ValueError(f"extractor worker crashed: {type(exc).__name__}") from exc
    except asyncio.TimeoutError as exc:
        _reset_extractor_executor(executor)
        raise ValueError(
            f"extraction exceeded {_EXTRACTION_TIMEOUT_SECONDS}s"
        ) from exc


def _clean_extracted_text(text: str | None) -> str:
    if not text:
        return ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_html_text(html: str, url: str) -> tuple[str, str, str]:
    """Extract article text, falling back from precision to recall."""
    soup = BeautifulSoup(html, "html.parser")
    title = str(soup.title.string).strip() if soup.title and soup.title.string else url

    extractors: list[tuple[str, str]] = []
    try:
        if len(html) > 750_000:
            primary_name = "trafilatura_baseline"
            primary_text = baseline(html)[1]
        else:
            primary_name = "trafilatura"
            primary_text = extract_main_text(
                html,
                url=url,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                favor_recall=True,
                fast=True,
            )
        extractors.append((primary_name, _clean_extracted_text(primary_text)))
    except Exception:
        extractors.append(("trafilatura", ""))

    try:
        extractors.append(("trafilatura_html2txt", _clean_extracted_text(html2txt(html))))
    except Exception:
        extractors.append(("trafilatura_html2txt", ""))

    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()
    extractors.append(("beautifulsoup", _clean_extracted_text(soup.get_text("\n"))))

    for extractor, text in extractors:
        if len(text) >= _MIN_EXTRACTED_TEXT_CHARS:
            return text, title, extractor
    extractor, text = max(extractors, key=lambda item: len(item[1]))
    return text, title, extractor


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from a text-bearing PDF held in memory."""
    pdf = None
    try:
        pdf = pdfium.PdfDocument(content)
        pages: list[str] = []
        for index in range(len(pdf)):
            page = pdf[index]
            text_page = None
            try:
                text_page = page.get_textpage()
                pages.append(text_page.get_text_bounded())
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()
        return _clean_extracted_text("\n\n".join(pages))
    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if pdf is not None:
            pdf.close()


@asynccontextmanager
async def shared_key_slot(key: str, concurrency: int, directory: Path | None = None):
    """Per-key limit shared by Dense/Sparse processes on this VM; no secrets logged."""
    root = directory or Path.home() / '.cache' / 'mtbench-serper-slots'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256(key.encode()).hexdigest()
    held = None
    try:
        while held is None:
            for index in range(concurrency):
                slot = (root / f'{digest}-{index}.lock').open('a')
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    slot.close()
                else:
                    held = slot
                    break
            if held is None:
                await asyncio.sleep(0.05)
        yield
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()


from .tokens import count_tokens as estimate_tokens, truncate_to_tokens  # noqa: E402


def document_length_metrics(text: str, limit: int | None) -> dict[str, Any]:
    used = text if limit is None else text[:limit]
    return dict(extracted_text_chars=len(text), retained_text_chars=len(used),
                extracted_nonwhitespace_chars=sum(not c.isspace() for c in text),
                retained_nonwhitespace_chars=sum(not c.isspace() for c in used),
                document_char_limit=limit, truncated=limit is not None and len(text) > limit,
                retained_fraction=len(used) / len(text) if text else 0.0)


class SerperPool:
    _PROXY_FIRST_DOMAINS = ("arxiv.org", "wikipedia.org", "wikimedia.org")

    def __init__(self, config: RetrievalConfig, trace: TraceWriter):
        self.config = config
        self.trace = trace
        self.keys = self._load_keys(config.keys_file)
        self._cycle = itertools.cycle(range(len(self.keys)))
        self._cycle_lock = asyncio.Lock()
        self._key_limits = [asyncio.Semaphore(config.per_key_concurrency) for _ in self.keys]
        self._direct_fetch_limit = asyncio.Semaphore(config.fetch_concurrency)
        self._proxy_fetch_limit = asyncio.Semaphore(min(config.fetch_concurrency, 8))
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LangGraphDeepResearchBench/1.0)"
        }
        search_proxy = config.proxy_url if config.search_via_proxy else None
        self._search_clients = [
            httpx.AsyncClient(
                follow_redirects=True,
                proxy=search_proxy,
                headers=headers,
                limits=httpx.Limits(
                    max_connections=config.per_key_concurrency,
                    max_keepalive_connections=config.per_key_concurrency,
                ),
            )
            for _ in self.keys
        ]
        fetch_limits = httpx.Limits(
            # The semaphore is the actual fetch-concurrency limit. Keep spare
            # connections so redirects and just-finished responses cannot
            # transiently starve the next admitted fetch.
            max_connections=max(config.fetch_concurrency * 2, 32),
            max_keepalive_connections=config.fetch_concurrency,
        )
        self._direct_fetch_client = httpx.AsyncClient(
            follow_redirects=True,
            headers=headers,
            limits=fetch_limits,
        )
        self._proxy_fetch_client = httpx.AsyncClient(
            follow_redirects=True,
            proxy=config.proxy_url,
            headers=headers,
            limits=httpx.Limits(
                max_connections=16,
                max_keepalive_connections=8,
            ),
        )

    @staticmethod
    def _load_keys(path: Path) -> list[str]:
        keys = [line.strip() for line in path.expanduser().read_text().splitlines()]
        keys = [key for key in keys if key and not key.startswith("#")]
        if not keys:
            raise RuntimeError(f"no Serper keys found in {path}")
        return keys

    async def close(self) -> None:
        await asyncio.gather(
            *(client.aclose() for client in self._search_clients),
            self._direct_fetch_client.aclose(),
            self._proxy_fetch_client.aclose(),
        )

    @classmethod
    def _proxy_first(cls, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith(f".{domain}") for domain in cls._PROXY_FIRST_DOMAINS)

    @staticmethod
    def _should_try_other_route(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 403 or status >= 500
        if isinstance(exc, ValueError):
            return str(exc) == "extracted page text was too short"
        return isinstance(exc, (httpx.RequestError, ssl.SSLError))

    async def _key_slot(self) -> tuple[str, asyncio.Semaphore, int]:
        async with self._cycle_lock:
            index = next(self._cycle)
        return self.keys[index], self._key_limits[index], index

    async def search(
        self, agent_id: str, query: str, page: int = 1
    ) -> list[dict[str, str]]:
        key, limit, key_index = await self._key_slot()
        started = time.perf_counter()
        await self.trace.emit(
            "search_started",
            agent_id=agent_id,
            query=query,
            page=page,
            key_index=key_index,
        )
        async with limit, shared_key_slot(key, self.config.per_key_concurrency):
            response = await self._search_clients[key_index].post(
                self.config.search_url,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={
                    "q": query,
                    "num": self.config.results_per_search,
                    "page": page,
                },
                timeout=self.config.search_timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        results = [
            {
                "url": str(item.get("link", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in payload.get("organic", [])
            if item.get("link")
        ]
        await self.trace.emit(
            "search_completed",
            agent_id=agent_id,
            query=query,
            page=page,
            result_count=len(results),
            key_index=key_index,
            elapsed_seconds=time.perf_counter() - started,
        )
        return results

    async def fetch(self, agent_id: str, result: dict[str, str], max_chars: int | None) -> Document | None:
        url = result["url"]
        if not _PDF_EXTRACTION_ENABLED and _is_pdf_url(url):
            await self.trace.emit(
                "fetch_skipped",
                agent_id=agent_id,
                url=url,
                reason="pdf_extraction_disabled",
            )
            return None
        direct = (
            "direct",
            self._direct_fetch_client,
            self._direct_fetch_limit,
            min(self.config.fetch_timeout_seconds, 5),
        )
        clash = (
            "clash",
            self._proxy_fetch_client,
            self._proxy_fetch_limit,
            self.config.fetch_timeout_seconds,
        )
        routes = (clash, direct) if self._proxy_first(url) else (direct, clash)
        for attempt, (route, client, limit, timeout) in enumerate(routes, start=1):
            started = time.perf_counter()
            await self.trace.emit(
                "fetch_started",
                agent_id=agent_id,
                url=url,
                attempt=attempt,
                route=route,
            )
            try:
                queued_at = time.perf_counter()
                async with limit:
                    admitted_at = time.perf_counter()
                    response = await client.get(url, timeout=timeout)
                downloaded_at = time.perf_counter()
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                content_type = response.headers.get("content-type", "").lower()
                if _is_pdf_response(url, content_type):
                    if not _PDF_EXTRACTION_ENABLED:
                        await self.trace.emit(
                            "fetch_skipped",
                            agent_id=agent_id,
                            url=url,
                            attempt=attempt,
                            route=route,
                            reason="pdf_extraction_disabled",
                            elapsed_seconds=time.perf_counter() - started,
                        )
                        return None
                    if len(response.content) > _MAX_PDF_BYTES:
                        raise ValueError("response body exceeded the 25 MB PDF limit")
                    text = await _run_isolated(_extract_pdf_text, response.content)
                    title = result.get("title") or url
                    extractor = "pypdfium2"
                else:
                    if len(response.content) > _MAX_HTML_BYTES:
                        raise ValueError("response body exceeded the 5 MB HTML limit")
                    text, inferred_title, extractor = await _run_isolated(
                        _extract_html_text, response.text, url
                    )
                    title = result.get("title") or inferred_title
                if len(text) < _MIN_EXTRACTED_TEXT_CHARS:
                    raise ValueError("extracted page text was too short")
                document = Document(
                    url=url,
                    title=title,
                    text=text if max_chars is None else text[:max_chars],
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                )
                await self.trace.emit(
                    "fetch_completed",
                    agent_id=agent_id,
                    url=url,
                    attempt=attempt,
                    route=route,
                    extractor=extractor,
                    text_chars=len(document.text),
                    **document_length_metrics(text, max_chars),
                    queue_seconds=admitted_at - queued_at,
                    download_seconds=downloaded_at - admitted_at,
                    extraction_seconds=time.perf_counter() - downloaded_at,
                    elapsed_seconds=document.elapsed_seconds,
                )
                return document
            except (httpx.HTTPError, ssl.SSLError, ValueError) as exc:
                await self.trace.emit(
                    "fetch_failed",
                    agent_id=agent_id,
                    url=url,
                    attempt=attempt,
                    route=route,
                    error=type(exc).__name__,
                    detail=str(exc)[:500],
                    elapsed_seconds=time.perf_counter() - started,
                )
                if attempt == len(routes) or not self._should_try_other_route(exc):
                    break
        return None

    async def collect(
        self,
        agent_id: str,
        query: str,
        count: int,
        max_chars: int | None,
        *,
        target_tokens: int | None = None,
        base_tokens: int = 0,
    ) -> list[Document]:
        """Assemble evidence up to a round budget.

        Search pages are requested in parallel groups: the Serper pool allows
        thirty keys times five concurrent calls, so the old page-by-page loop
        left almost all of that idle.  Every document that is downloaded is
        used, and only the single document that crosses the budget is cut, so
        no fetched page is discarded and no page except the last is truncated.
        """
        seen: set[str] = set()
        documents: list[Document] = []
        candidates: list[dict[str, str]] = []
        candidate_pages = 0
        pages_searched = 0
        used_tokens = base_tokens
        budget_reached = False

        def remaining_tokens() -> int | None:
            return None if target_tokens is None else target_tokens - used_tokens

        page = 1
        while page <= self.config.max_search_pages and not budget_reached:
            # Speculative page prefetch only pays off when a token budget is the
            # binding constraint; with a plain document count we would burn
            # Serper calls on pages we are about to stop needing.
            if target_tokens is None or used_tokens >= target_tokens * 0.6:
                width = 1
            else:
                width = _SEARCH_PAGE_BATCH
            group = list(range(page, min(page + width, self.config.max_search_pages + 1)))
            page = group[-1] + 1
            pages = await asyncio.gather(
                *(self.search(agent_id, query, page=n) for n in group),
                return_exceptions=True,
            )
            for n, results in zip(group, pages):
                if isinstance(results, BaseException):
                    if isinstance(results, (httpx.HTTPError, ssl.SSLError)):
                        await self.trace.emit(
                            "search_failed", agent_id=agent_id, query=query, page=n,
                            error=type(results).__name__, detail=str(results)[:500],
                        )
                        continue
                    raise results
                pages_searched = max(pages_searched, n)
                fresh = [r for r in results if r["url"] not in seen]
                seen.update(r["url"] for r in fresh)
                candidates.extend(fresh)
                candidate_pages += len(fresh)

            cursor = 0
            while cursor < len(candidates) and len(documents) < count and not budget_reached:
                batch = candidates[cursor : cursor + min(count - len(documents), _FETCH_BATCH_SIZE)]
                cursor += len(batch)
                fetched = await asyncio.gather(
                    *(self.fetch(agent_id, result, max_chars) for result in batch)
                )
                for document in fetched:
                    if document is None:
                        continue
                    left = remaining_tokens()
                    if left is not None and left <= 0:
                        budget_reached = True
                        break
                    tokens = estimate_tokens(document.text)
                    if left is not None and tokens > left:
                        # Only this last document is cut, and only to the exact
                        # remaining budget; every earlier one stays whole.
                        clipped = truncate_to_tokens(document.text, left)
                        await self.trace.emit(
                            "document_budget_truncated", agent_id=agent_id,
                            url=document.url, original_chars=len(document.text),
                            retained_chars=len(clipped), remaining_tokens=left,
                        )
                        document = Document(
                            url=document.url, title=document.title, text=clipped,
                            fetched_at=document.fetched_at,
                            elapsed_seconds=document.elapsed_seconds,
                        )
                        tokens = estimate_tokens(clipped) if clipped else 0
                        budget_reached = True
                    if document.text:
                        documents.append(document)
                        used_tokens += tokens
                    if left is not None and used_tokens >= target_tokens:
                        budget_reached = True
                        break
            candidates = candidates[cursor:]

            await self.trace.emit(
                "retrieval_page_completed", agent_id=agent_id, query=query,
                page=pages_searched, new_candidates=candidate_pages,
                successful_documents=len(documents), requested_documents=count,
                estimated_tokens=used_tokens,
            )
            if len(documents) >= count:
                break

        await self.trace.emit(
            "retrieval_completed", agent_id=agent_id, requested_documents=count,
            successful_documents=len(documents), candidate_pages=candidate_pages,
            pages_searched=pages_searched, target_tokens=target_tokens,
            base_tokens=base_tokens, estimated_tokens=used_tokens,
        )
        return documents


    async def expand_document(
        self,
        agent_id: str,
        document: Document,
        max_chars: int,
        document_id: str,
    ) -> Document:
        """Refetch one round-one source with a larger text budget for close reading."""
        expanded = await self.fetch(
            agent_id,
            {"url": document.url, "title": document.title, "snippet": ""},
            max_chars,
        )
        if expanded is None:
            await self.trace.emit(
                "document_expansion_failed",
                agent_id=agent_id,
                document_id=document_id,
                url=document.url,
                fallback_chars=len(document.text),
            )
            return document
        await self.trace.emit(
            "document_expansion_completed",
            agent_id=agent_id,
            document_id=document_id,
            url=document.url,
            original_chars=len(document.text),
            expanded_chars=len(expanded.text),
        )
        return expanded
