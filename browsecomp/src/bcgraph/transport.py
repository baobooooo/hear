"""OpenAI-compatible HTTP without library-side retries or implicit prompt rewriting.

The Sparse-vLLM extension is used only on the configured H2O endpoint. An uncertain
append is never replayed against the same chain. Successful raw responses are
journaled before parsing model-generated JSON.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
import time
from typing import Any
import uuid
import httpx

from .admission import PriorityGate
from .config import EndpointConfig
from .evidence import stable_hash
from .storage import Store


class ModelRequestError(RuntimeError):
    pass


class ModelHTTPError(ModelRequestError):
    def __init__(self, status: int, code: str | None, message: str):
        super().__init__(f"HTTP {status} [{code or 'no-code'}]: {message}")
        self.status, self.code = status, code


class AmbiguousCommitError(ModelRequestError):
    pass


class CacheProtocolError(ModelRequestError):
    pass


def completion_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def chain_invalidate_url(completions_url: str) -> str:
    base = completions_url.rsplit("/chat/completions", 1)[0].rstrip("/")
    return base + "/chain_cache/invalidate"


def observed_usage(raw: dict | None) -> dict:
    raw = raw or {}
    warnings = []
    def integer(value: Any, name: str):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            warnings.append(f"Invalid {name}")
            return None
        return value
    prompt = integer(raw.get("prompt_tokens"), "prompt_tokens")
    output = integer(raw.get("completion_tokens"), "completion_tokens")
    reused = integer(raw.get("reused_tokens"), "reused_tokens")
    source = "sparse.reused_tokens" if reused is not None else "unknown"
    if reused is None:
        details = raw.get("prompt_tokens_details") or {}
        reused = integer(details.get("cached_tokens"), "prompt_tokens_details.cached_tokens")
        if reused is not None:
            source = "prompt_tokens_details.cached_tokens"
    if reused is None:
        reused = integer(raw.get("cached_tokens"), "cached_tokens")
        if reused is not None:
            source = "usage.cached_tokens"
    if reused is None:
        reused = integer(raw.get("prompt_cache_hit_tokens"), "prompt_cache_hit_tokens")
        if reused is not None:
            source = "deepseek.prompt_cache_hit_tokens"
    prefilled = integer(raw.get("prefilled_tokens"), "prefilled_tokens")
    prefill_source = "explicit" if prefilled is not None else "unknown"
    if prompt is not None and reused is not None and reused > prompt:
        warnings.append("reused_tokens exceeds reported prompt_tokens; semantics are not comparable")
        reused, source = None, "inconsistent"
    if prefilled is None and prompt is not None and reused is not None:
        prefilled = prompt - reused
        prefill_source = "derived_prompt_minus_cached"
    if prefilled is not None and prompt is not None and prefilled > prompt:
        warnings.append("prefilled_tokens exceeds prompt_tokens")
        prefilled, prefill_source = None, "inconsistent"
    if (prefill_source == "explicit" and prompt is not None and reused is not None
            and prefilled + reused != prompt):
        warnings.append("explicit prefill/reuse counters do not sum to prompt_tokens")
    reasoning = integer((raw.get("completion_tokens_details") or {}).get("reasoning_tokens"), "reasoning_tokens")
    return {"reported_prompt_tokens": prompt, "completion_tokens": output,
            "reused_tokens": reused, "prefilled_tokens": prefilled,
            "reuse_counter_source": source, "prefill_counter_source": prefill_source,
            "reasoning_tokens": reasoning, "warnings": warnings}


def observed_server_timing(raw: Any, *, engine: str, output_tokens: int | None) -> dict:
    """Keep installed vLLM's scheduled-to-first field distinct from API TTFT.

    SparseEngine's extension uses explicit names and a schema marker;
    similarly named unknown extensions must not silently acquire these meanings.
    """
    fields = ("server_ttft_ms", "server_queue_ms", "server_prefill_ms",
              "server_queue_to_first_token_ms", "server_decode_ms", "server_tpot_ms",
              "server_decode_tokens_per_second", "server_inference_tokens_per_second")
    result = {key: None for key in fields}
    result.update(source="unavailable", warnings=[])
    if raw is None:
        return result
    if not isinstance(raw, dict):
        result["warnings"].append("Server metrics is not an object")
        return result

    def number(key):
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            result["warnings"].append(f"Invalid server metric {key}")
            return None
        return float(value)

    if engine == "vllm":
        result.update(source="vllm.per_request_metrics.scheduled_to_first.v1",
                      server_queue_ms=number("queue_time_ms"),
                      server_prefill_ms=number("time_to_first_token_ms"),
                      server_decode_ms=number("generation_time_ms"),
                      server_inference_tokens_per_second=number("tokens_per_second"))
        # Do not call mean_itl_ms a measured distribution of token intervals.
        reported_tpot = number("mean_itl_ms")
        result["reported_mean_itl_ms"] = reported_tpot
        queue, prefill = result["server_queue_ms"], result["server_prefill_ms"]
        if queue is not None and prefill is not None:
            result["server_queue_to_first_token_ms"] = queue + prefill
    elif engine == "sparse-vllm" and raw.get("schema") == "bcgraph.server_timing.v1":
        result["source"] = "sparseengine.dispatcher.v1"
        for key in fields:
            result[key] = number(key)
        result["clock_basis"] = raw.get("clock_basis")
        for key in ("server_first_token_event_ms", "dispatcher_prepare_ms", "dispatcher_admission_wait_ms",
                    "observed_output_tokens", "token_event_count", "coalesced_token_events"):
            result[key] = number(key)
        for key in ("prefill_steps", "decode_steps"):
            result[key] = number(key)
        complete = raw.get("token_timing_complete") is True
        intervals = raw.get("token_itl_ms")
        if complete and isinstance(intervals, list) and isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
            valid = len(intervals) == max(0, output_tokens - 1) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in intervals)
        else:
            valid = False
        result["token_timing_complete"] = complete and valid
        result["token_itl_ms"] = intervals if valid else None
        if not result["token_timing_complete"]:
            result["server_ttft_ms"] = result["server_decode_ms"] = None
    else:
        result["warnings"].append("Unknown server timing schema")
        return result
    decode = result["server_decode_ms"]
    valid_count = isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens > 1
    if valid_count and decode is not None:
        result["server_tpot_ms"] = decode / (output_tokens - 1)
        result["server_decode_tokens_per_second"] = (output_tokens - 1) * 1000 / decode if decode > 0 else None
    else:
        result["server_tpot_ms"] = result["server_decode_tokens_per_second"] = None
    return result


def _error(response: httpx.Response) -> ModelHTTPError:
    try:
        payload = response.json()
    except ValueError:
        return ModelHTTPError(response.status_code, None, response.text[:1000])
    detail = payload.get("error", payload.get("detail", payload)) if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return ModelHTTPError(response.status_code, detail.get("code"),
                              str(detail.get("message", detail))[:1000])
    return ModelHTTPError(response.status_code, None, str(detail)[:1000])


def request_message(message: dict) -> dict:
    """Preserve generated fields verbatim; omit response-only transport metadata."""
    allowed = {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"}
    cleaned = {k: deepcopy(v) for k, v in message.items() if k in allowed}
    if "reasoning_content" not in cleaned and isinstance(message.get("reasoning"), str):
        cleaned["reasoning_content"] = message["reasoning"]
    return cleaned


class ChatClient:
    def __init__(self, config: EndpointConfig, store: Store, *,
                 gate: PriorityGate | None = None, http: httpx.AsyncClient | None = None,
                 store_raw: bool = True):
        self.config, self.store = config, store
        self.gate = gate or PriorityGate(config.max_inflight)
        self.http = http or httpx.AsyncClient(timeout=config.timeout_seconds, trust_env=False)
        self._owns_http = http is None
        self.url = completion_url(config.base_url)
        self.client_epoch = uuid.uuid4().hex
        self._writers: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.store_raw = store_raw

    def build_payload(self, messages: list[dict], max_tokens: int,
                      handle: dict | None, *, enable_thinking: bool | None = None) -> tuple[dict, str]:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Expected a nonempty full transcript ending in a user message")
        cfg = self.config
        thinking = cfg.enable_thinking if enable_thinking is None else enable_thinking
        payload = {"model": cfg.model, "messages": [request_message(m) for m in messages],
                   "max_tokens": max_tokens, "stream": False,
                   "temperature": cfg.temperature, "top_p": cfg.top_p, **cfg.extra_body}
        kwargs = dict(cfg.chat_template_kwargs)
        if cfg.engine == "deepseek":
            if thinking is not None:
                payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        elif thinking is not None:
            kwargs["enable_thinking"] = thinking
            if cfg.engine == "sparse-vllm":
                payload["enable_thinking"] = thinking
        if cfg.preserve_thinking is not None:
            kwargs["preserve_thinking"] = cfg.preserve_thinking
            if cfg.engine == "sparse-vllm":
                payload["preserve_thinking"] = cfg.preserve_thinking
        if kwargs:
            payload["chat_template_kwargs"] = kwargs
        mode = "full_prefix" if cfg.cache == "prefix" else "cold_full"
        if cfg.cache != "chain":
            return payload, mode
        valid = bool(handle and handle.get("chain_id") and handle.get("endpoint") == self.url
                     and handle.get("engine_epoch") == cfg.engine_epoch
                     and handle.get("client_epoch") == self.client_epoch)
        if not valid:
            return payload, "cold_stale_handle" if handle else "cold_full"
        if handle.get("history_hash") != stable_hash(messages[:-1]):
            raise CacheProtocolError("Local committed history changed while retaining a chain handle")
        if len(messages) < 3 or messages[-2].get("role") != "assistant":
            raise CacheProtocolError("Chain append must follow a committed assistant message")
        payload["chain_id"] = handle["chain_id"]
        if cfg.chain_transport == "delta":
            payload["messages"] = [request_message(m) for m in messages[-2:]]
            payload["chain_append_start"] = 1
            return payload, "chain_delta"
        return payload, "chain_full"

    async def complete(self, messages: list[dict], max_tokens: int, *, operation_id: str,
                       writer_key: str, handle: dict | None = None,
                       continuation: bool = False, local_prompt_tokens: int | None = None,
                       enable_thinking: bool | None = None) -> dict:
        # Serializes cold starts as well as appends for one logical cell.
        async with self._writers[writer_key]:
            return await self._complete(messages, max_tokens, operation_id, handle,
                                        continuation, local_prompt_tokens, enable_thinking)

    async def _complete(self, messages: list[dict], max_tokens: int, operation_id: str,
                        handle: dict | None, continuation: bool, local_prompt_tokens: int | None,
                        enable_thinking: bool | None):
        cfg = self.config
        request = {"config": cfg.model_dump(), "messages": messages, "max_tokens": max_tokens}
        if enable_thinking is not None:
            request["enable_thinking"] = enable_thinking
        fingerprint = stable_hash(request)
        entry = self.store.get(operation_id, fingerprint)
        if entry:
            if entry["status"] == "success":
                result = deepcopy(entry["value"])
                result["journal_replay"] = True
                self.store.event("model_journal_replay", operation_id=operation_id)
                return result
            if entry["status"] in {"pending", "ambiguous"}:
                raise AmbiguousCommitError(
                    f"Uncertain earlier request {operation_id}; use a new query attempt, not the old chain")
        payload, mode = self.build_payload(messages, max_tokens, handle, enable_thinking=enable_thinking)
        if self.store_raw:
            # Separate namespace: observation is not a completed model call.
            # Persist before HTTP so failed/ambiguous requests remain replayable
            # as offline workloads rather than disappearing from selection.
            self.store.put('request-input:' + operation_id, fingerprint, 'success', {
                'messages': deepcopy(messages), 'max_tokens': max_tokens,
                'model': cfg.model, 'temperature': cfg.temperature, 'top_p': cfg.top_p,
                'enable_thinking': cfg.enable_thinking if enable_thinking is None else enable_thinking,
                'logical_prompt_tokens': local_prompt_tokens})
        recovered, retries, attempts = 0, 0, 0
        total_start = time.monotonic()
        while True:
            attempts += 1
            self.store.put(operation_id, fingerprint, "pending", {"mode": mode, "attempt": attempts})
            started = time.monotonic()
            wait_ms = 0.0
            try:
                async with self.gate.slot(continuation=continuation) as wait_ms:
                    request_start = time.monotonic()
                    admission_snapshot = self.gate.snapshot()
                    headers = {"Content-Type": "application/json",
                               "X-Request-ID": stable_hash([operation_id, attempts])[:32],
                               "Connection": "close"}
                    # Uvicorn and httpx both default to a short keep-alive timeout.
                    # Long tool intervals can race the server closing an idle socket,
                    # which makes an otherwise successful stateful append ambiguous.
                    key = os.environ.get(cfg.api_key_env)
                    if key:
                        headers["Authorization"] = f"Bearer {key}"
                    response = await self.http.post(self.url, json=payload, headers=headers)
                http_ms = (time.monotonic() - request_start) * 1000
                if response.status_code >= 400:
                    raise _error(response)
                raw = response.json()
                choice = raw["choices"][0]
                raw_assistant = choice["message"]
                if raw_assistant.get("role", "assistant") != "assistant":
                    raise CacheProtocolError("Endpoint did not return an assistant message")
                assistant = request_message({"role": "assistant", **raw_assistant})
                usage = observed_usage(raw.get("usage"))
                server_timing = observed_server_timing(
                    raw.get("metrics"), engine=cfg.engine,
                    output_tokens=usage["completion_tokens"])
                chain_id = raw.get("chain_id") or response.headers.get("X-SparseVLLM-Chain-ID")
                next_handle = None
                if cfg.cache == "chain" and chain_id:
                    next_handle = {"chain_id": chain_id, "endpoint": self.url,
                                   "engine_epoch": cfg.engine_epoch, "client_epoch": self.client_epoch,
                                   "history_hash": stable_hash([*messages, assistant])}
                result = {"assistant": assistant, "raw_assistant": raw_assistant,
                          "enable_thinking": payload.get("chat_template_kwargs", {}).get("enable_thinking",
                              cfg.enable_thinking if enable_thinking is None else enable_thinking),
                          "handle": next_handle, "usage": usage,
                          "finish_reason": choice.get("finish_reason"),
                          "chain_status": raw.get("chain_status"), "request_mode": mode,
                          "attempts": attempts, "recoveries": recovered, "journal_replay": False,
                          "duration_ms": (time.monotonic() - total_start) * 1000}
                result.update(http_ms=http_ms, admission_wait_ms=wait_ms,
                              admission_snapshot=admission_snapshot, server_timing=server_timing,
                              client_request_id=headers["X-Request-ID"], server_request_id=raw.get("id"))
                if cfg.cache == "chain" and not chain_id:
                    result["cache_warning"] = "chain_id missing; server did not establish a reusable chain"
                if self.store_raw:
                    result["wire_payload"] = payload
                    result["raw_response"] = raw
                # Durable commit precedes all model-content parsing/validation.
                self.store.put(operation_id, fingerprint, "success", result)
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 engine=cfg.engine, method=cfg.method, cache=cfg.cache,
                                 attempt=attempts, success=True, request_mode=mode,
                                 http_ms=http_ms, admission_wait_ms=wait_ms,
                                 admission_snapshot=admission_snapshot,
                                 local_logical_prompt_tokens=local_prompt_tokens,
                                 enable_thinking=result["enable_thinking"], max_tokens=max_tokens,
                                 sent_bytes=len(json.dumps(payload, ensure_ascii=False).encode()),
                                 usage=usage, finish_reason=choice.get("finish_reason"),
                                 chain_status=raw.get("chain_status"), chain_id_present=bool(chain_id),
                                 ttft_ms=None, server_timing=server_timing,
                                 client_request_id=headers["X-Request-ID"], server_request_id=raw.get("id"))
                return result
            except asyncio.CancelledError:
                self.store.put(operation_id, fingerprint, "ambiguous", {"error": "cancelled"})
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 success=False, request_mode=mode, attempt=attempts,
                                 error="cancelled", usage_unknown=True,
                                 elapsed_ms=(time.monotonic() - started) * 1000)
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                # These transport failures occur before a request is sent.
                self.store.put(operation_id, fingerprint, "rejected", {"error": str(exc)})
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 success=False, request_mode=mode, attempt=attempts,
                                 error=type(exc).__name__, usage_unknown=False,
                                 elapsed_ms=(time.monotonic() - started) * 1000)
                if retries >= cfg.max_safe_retries:
                    raise ModelRequestError(str(exc)) from exc
                retries += 1
                await asyncio.sleep(0.3 * retries)
            except httpx.ReadError as exc:
                # A response may have been lost after inference committed. A
                # stateless prefix request can be replayed without mutating any
                # server-side conversation. Chain requests cannot: replaying a
                # create/append could fork or advance the chain twice.
                replayable = cfg.cache != "chain"
                self.store.put(operation_id, fingerprint,
                               "pending" if replayable and retries < cfg.max_safe_retries else "ambiguous",
                               {"error": str(exc), "attempt": attempts})
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 success=False, request_mode=mode, attempt=attempts,
                                 error=type(exc).__name__, usage_unknown=True,
                                 safe_replay=replayable,
                                 elapsed_ms=(time.monotonic() - started) * 1000)
                if replayable and retries < cfg.max_safe_retries:
                    retries += 1
                    await asyncio.sleep(0.3 * retries)
                    continue
                raise AmbiguousCommitError(
                    f"Request may have committed; not retrying append: {type(exc).__name__}: {exc}") from exc

            except ModelHTTPError as exc:
                known_missing = (exc.status in (404, 410) and exc.code in cfg.chain_missing_codes
                                 and "chain_id" in payload)
                rejected = known_missing or exc.status in (400, 401, 403, 404, 409, 422, 429) or exc.code == "chain_capacity_unavailable"
                self.store.put(operation_id, fingerprint, "rejected" if rejected else "ambiguous",
                               {"error": str(exc)})
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 success=False, request_mode=mode, attempt=attempts,
                                 error=str(exc), code=exc.code, status=exc.status,
                                 usage_unknown=not rejected,
                                 elapsed_ms=(time.monotonic() - started) * 1000)
                if known_missing and recovered < cfg.max_cold_recoveries:
                    recovered += 1
                    payload, mode = self.build_payload(messages, max_tokens, None, enable_thinking=enable_thinking)
                    mode = "cold_recovery"
                    continue
                if (exc.status == 429 or exc.code == "chain_capacity_unavailable") and retries < cfg.max_safe_retries:
                    retries += 1
                    await asyncio.sleep(0.3 * retries)
                    continue
                # Busy/prefix/fingerprint mismatches are not masked as cache misses.
                raise
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, CacheProtocolError) as exc:
                self.store.put(operation_id, fingerprint, "ambiguous", {"error": str(exc)})
                self.store.event("model_attempt", operation_id=operation_id, endpoint=self.url,
                                 success=False, request_mode=mode, attempt=attempts,
                                 error=str(exc), usage_unknown=True,
                                 elapsed_ms=(time.monotonic() - started) * 1000)
                raise AmbiguousCommitError(
                    f"Request may have committed; not retrying append: {type(exc).__name__}: {exc}") from exc

    async def release_chain(self, handle: dict) -> dict:
        """Release a resident chain after its logical research cell is permanently done."""
        chain_id = str(handle.get("chain_id") or "").strip()
        if not chain_id or handle.get("endpoint") != self.url:
            raise CacheProtocolError("Cannot release a missing chain or a chain owned by another endpoint")
        response = await self.http.post(
            chain_invalidate_url(self.url),
            json={"chain_id": chain_id},
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        if response.status_code in (404, 410):
            error = _error(response)
            if error.code in self.config.chain_missing_codes:
                return {"chain_id": chain_id, "released": False, "already_absent": True}
            raise error
        if response.status_code >= 400:
            raise _error(response)
        payload = response.json()
        return {"chain_id": chain_id, "released": True, "result": payload}

    async def close(self):
        if self._owns_http:
            await self.http.aclose()
