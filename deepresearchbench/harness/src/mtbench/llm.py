from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import ModelConfig
from .prefill_gate import PrefillGate
from .tokens import count_tokens
from .trajectory import TraceWriter

T = TypeVar("T", bound=BaseModel)

CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (?P<context>\d+) tokens.*?"
    r"prompt contains at least (?P<prompt>\d+) input tokens",
    re.DOTALL,
)
SPARSE_CONTEXT_LIMIT_RE = re.compile(
    r"Prompt length \+ max_tokens exceeds max_model_len:\s*"
    r"(?P<prompt>\d+)\s*\+\s*\d+\s*>\s*(?P<context>\d+)"
)
CONTEXT_RETRY_SAFETY_TOKENS = 64
CONTEXT_COMPACTION_MARGIN = 0.97
CONTEXT_ELISION = "\n[...context compacted...]\n"
CONNECTION_RETRY_LIMIT = 20
CONNECTION_RETRY_DELAY_SECONDS = 2.0
# Sparse-vLLM answers 503 chain_capacity_unavailable when every chain row is
# pinned by an in-flight request.  That is congestion, not a failure: the dense
# baseline meets the same congestion by queueing (its logs show
# "Running: 1 reqs, Waiting: 15 reqs" with zero preemptions).  Treating it as
# fatal made three of four workflows die while the first one to grab capacity
# ran to completion, which measures scheduling luck rather than the engine.
CHAIN_CAPACITY_RETRY_LIMIT = 120
CHAIN_CAPACITY_RETRY_DELAY_SECONDS = 5.0
# Room a resumed chain already holds on the server (SnapKV compacts to about
# sink + decode_keep + recent = 4.7k slots, plus the previous round's output).
PREFILL_GATE_CHAIN_TOKENS = int(os.environ.get("MTBENCH_PREFILL_GATE_CHAIN_TOKENS", "6000"))
PREFILL_GATE_MESSAGE_OVERHEAD_TOKENS = 8


def prefill_reservation_tokens(args: dict[str, Any]) -> int:
    """Slots the server reserves for this request's prefill.

    A chain append prefills only the suffix on top of the live chain; any other
    request (including a 404/410 rebuild) prefills the whole transcript.
    """
    messages = args["messages"]
    start = (args.get("extra_body") or {}).get("chain_append_start")
    sent = messages[start:] if start is not None else messages
    tokens = sum(
        count_tokens(message["content"]) + PREFILL_GATE_MESSAGE_OVERHEAD_TOKENS
        for message in sent
    )
    return tokens + (PREFILL_GATE_CHAIN_TOKENS if start is not None else 0)


def is_chain_rebuild_error(exc: Exception) -> bool:
    """Return whether the server no longer retains a requested chain."""
    if getattr(exc, "status_code", None) not in {404, 410}:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("detail", body)
    return isinstance(detail, dict) and detail.get("code") in {
        "chain_gone",
        "chain_not_found",
    }


def is_chain_capacity_error(exc: Exception) -> bool:
    """Return whether the server is momentarily out of chain rows."""
    if getattr(exc, "status_code", None) != 503:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("detail", body)
    return (
        isinstance(detail, dict)
        and detail.get("code") == "chain_capacity_unavailable"
    )


def context_window_sizes(exc: Exception) -> tuple[int, int] | None:
    if getattr(exc, "status_code", None) != 400:
        return None
    body = getattr(exc, "body", None)
    message = json.dumps(body) if isinstance(body, dict) else str(exc)
    match = CONTEXT_LIMIT_RE.search(message) or SPARSE_CONTEXT_LIMIT_RE.search(message)
    if match is None:
        return None
    return int(match.group("prompt")), int(match.group("context"))


def context_retry_max_tokens(exc: Exception, requested_max_tokens: int) -> int | None:
    sizes = context_window_sizes(exc)
    if sizes is None:
        return None
    prompt_tokens, context_tokens = sizes
    # "At least" can be a tokenizer scan lower bound rather than the final
    # prompt size. Exponential backoff avoids chasing that moving boundary.
    available = context_tokens - prompt_tokens - CONTEXT_RETRY_SAFETY_TOKENS
    if available < 1 or requested_max_tokens <= 1:
        return None
    return min(available, max(1, requested_max_tokens // 2))


def _compact_text_blocks(text: str, target_chars: int) -> str:
    if target_chars >= len(text):
        return text
    if target_chars <= 0:
        return ""
    blocks = text.split("\n\n")
    separator_chars = min(2 * max(0, len(blocks) - 1), max(0, target_chars // 10))
    content_budget = max(1, target_chars - separator_chars)
    total_content = sum(len(block) for block in blocks) or 1
    allocations = [int(content_budget * len(block) / total_content) for block in blocks]
    remainder = content_budget - sum(allocations)
    for index in sorted(range(len(blocks)), key=lambda i: len(blocks[i]), reverse=True):
        if remainder <= 0:
            break
        allocations[index] += 1
        remainder -= 1

    compacted: list[str] = []
    for block, allocation in zip(blocks, allocations, strict=True):
        if len(block) <= allocation:
            compacted.append(block)
            continue
        if allocation <= len(CONTEXT_ELISION) + 2:
            compacted.append(block[:allocation])
            continue
        visible = allocation - len(CONTEXT_ELISION)
        head = max(1, int(visible * 0.65))
        tail = max(1, visible - head)
        compacted.append(block[:head] + CONTEXT_ELISION + block[-tail:])
    result = "\n\n".join(compacted)
    return result[:target_chars]


def compact_messages_for_context(
    messages: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    context_tokens: int,
    requested_max_tokens: int,
) -> tuple[list[dict[str, Any]], int, int] | None:
    target_prompt_tokens = context_tokens - requested_max_tokens - CONTEXT_RETRY_SAFETY_TOKENS
    if prompt_tokens <= 0 or target_prompt_tokens <= 0 or target_prompt_tokens >= prompt_tokens:
        return None
    shrinkable = [
        index
        for index, message in enumerate(messages)
        if message.get("role") != "system" and isinstance(message.get("content"), str)
    ]
    before_chars = sum(len(str(message.get("content", ""))) for message in messages)
    shrinkable_chars = sum(len(messages[index]["content"]) for index in shrinkable)
    if not shrinkable or shrinkable_chars <= 0:
        return None
    ratio = max(
        0.05,
        min(0.99, target_prompt_tokens / prompt_tokens * CONTEXT_COMPACTION_MARGIN),
    )
    compacted = [dict(message) for message in messages]
    for index in shrinkable:
        content = messages[index]["content"]
        compacted[index]["content"] = _compact_text_blocks(
            content, max(64, int(len(content) * ratio))
        )
    after_chars = sum(len(str(message.get("content", ""))) for message in compacted)
    if after_chars >= before_chars:
        return None
    return compacted, before_chars, after_chars


def extract_json_candidates(text: str) -> list[Any]:
    """Return every decodable JSON payload in model output, outer first."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            candidates.append(value)
        except json.JSONDecodeError:
            continue
    if not candidates:
        raise ValueError("model output did not contain a JSON object")
    return candidates


def extract_json(text: str) -> Any:
    candidates = extract_json_candidates(text)
    # A model may put a small JSON example before its actual fenced response.
    # Preserve the historical standalone helper behavior by selecting the
    # largest outer payload; structured() below validates every candidate.
    return max(
        candidates,
        key=lambda candidate: len(json.dumps(candidate, ensure_ascii=False)),
    )


class LLMEndpoint:
    def __init__(self, role: str, config: ModelConfig, trace: TraceWriter):
        self.role = role
        self.config = config
        self.trace = trace
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self.semaphore = (
            asyncio.Semaphore(config.max_concurrency)
            if config.max_concurrency is not None
            else None
        )
        self.prefill_gate = PrefillGate.from_env(role)
        self.request_gate = PrefillGate.request_gate_from_env(role)

    async def text(
        self,
        stage: str,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        chain_id: str | None = None,
        chain_append_start: int | None = None,
        messages: list[dict[str, str]] | None = None,
        recovery_messages: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        request_messages = messages or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if chain_append_start is not None:
            if not chain_id:
                raise ValueError("chain_append_start requires chain_id")
            if not 1 <= chain_append_start < len(request_messages):
                raise ValueError("chain_append_start must split a non-empty suffix")
            if request_messages[chain_append_start - 1].get("role") != "assistant":
                raise ValueError("chain append suffix must begin after an assistant memo")
            if not recovery_messages:
                raise ValueError("chain append requires full messages for 404/410 recovery")
        queue_started = time.perf_counter()
        if self.semaphore is not None:
            await self.semaphore.acquire()
            await self.trace.emit(
                "llm_queue_completed",
                role=self.role,
                stage=stage,
                elapsed_seconds=time.perf_counter() - queue_started,
            )
        try:
            started = time.perf_counter()
            await self.trace.emit(
                "llm_request_started",
                role=self.role,
                stage=stage,
                endpoint=self.config.base_url,
                model=self.config.model,
                input_chars=sum(len(message["content"]) for message in request_messages),
            )
            if chain_append_start is not None:
                await self.trace.emit(
                    "chain_append_request_started",
                    role=self.role,
                    stage=stage,
                    requested_chain_id=chain_id,
                    chain_append_start=chain_append_start,
                    sent_message_count=len(request_messages),
                    sent_chars=sum(len(message["content"]) for message in request_messages),
                    recovery_message_count=len(recovery_messages or []),
                    recovery_chars=sum(
                        len(message["content"]) for message in recovery_messages or []
                    ),
                )
            request_args = {
                "model": self.config.model,
                "messages": request_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {
                        "enable_thinking": self.config.enable_thinking
                    },
                    **({"chain_id": chain_id} if chain_id else {}),
                    **(
                        {"chain_append_start": chain_append_start}
                        if chain_append_start is not None
                        else {}
                    ),
                },
            }

            async def consume_stream(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                if self.request_gate is not None:
                    slot = await self.request_gate.acquire(1)
                    await self.trace.emit(
                        "request_gate_acquired",
                        role=self.role,
                        stage=stage,
                        waited_seconds=slot.waited_seconds,
                        in_flight=slot.in_flight_tokens,
                    )
                    try:
                        return await consume_stream_gated(args)
                    finally:
                        await self.request_gate.release(slot)
                return await consume_stream_gated(args)

            async def consume_stream_gated(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                grant = None
                if self.prefill_gate is not None:
                    need = await asyncio.to_thread(prefill_reservation_tokens, args)
                    grant = await self.prefill_gate.acquire(need)
                    await self.trace.emit(
                        "prefill_gate_acquired",
                        role=self.role,
                        stage=stage,
                        tokens=grant.tokens,
                        waited_seconds=grant.waited_seconds,
                        in_flight_tokens=grant.in_flight_tokens,
                    )
                try:
                    return await consume_stream_ungated(args, grant)
                finally:
                    if grant is not None:
                        await self.prefill_gate.release(grant)

            async def consume_stream_ungated(
                args: dict[str, Any], grant: Any = None,
            ) -> tuple[str, dict[str, Any]]:
                stream = await self.client.chat.completions.create(**args)
                parts: list[str] = []
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason: str | None = None
                response_chain_id: str | None = None
                chain_status: str | None = None
                reasoning_chars = 0
                reused_tokens = 0
                prefilled_tokens = 0
                ttft_seconds: float | None = None
                async for chunk in stream:
                    extra = chunk.model_extra or {}
                    response_chain_id = extra.get("chain_id") or response_chain_id
                    chain_status = extra.get("chain_status") or chain_status
                    if chunk.usage is not None:
                        prompt_tokens = int(chunk.usage.prompt_tokens or 0)
                        completion_tokens = int(chunk.usage.completion_tokens or 0)
                        usage_extra = getattr(chunk.usage, "model_extra", None) or {}
                        reused_tokens = int(usage_extra.get("reused_tokens") or reused_tokens)
                        prefilled_tokens = int(
                            usage_extra.get("prefilled_tokens") or prefilled_tokens
                        )
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    reasoning = getattr(choice.delta, "reasoning_content", None) or (
                        (choice.delta.model_extra or {}).get("reasoning_content")
                        if hasattr(choice.delta, "model_extra") else None
                    )
                    if reasoning:
                        if ttft_seconds is None:
                            ttft_seconds = time.perf_counter() - started
                            if grant is not None:
                                await self.prefill_gate.release(grant)
                                grant = None
                        reasoning_chars += len(reasoning)
                    part = choice.delta.content or ""
                    if part:
                        if ttft_seconds is None:
                            ttft_seconds = time.perf_counter() - started
                            # The first token means the prefill reservation is cleared.
                            if grant is not None:
                                await self.prefill_gate.release(grant)
                                grant = None
                        parts.append(part)
                elapsed_seconds = time.perf_counter() - started
                return "".join(parts), {
                    "elapsed_seconds": elapsed_seconds,
                    "ttft_seconds": ttft_seconds if ttft_seconds is not None else elapsed_seconds,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "chain_id": response_chain_id,
                    "chain_status": chain_status,
                    "reused_tokens": reused_tokens,
                    "prefilled_tokens": prefilled_tokens,
                    "reasoning_chars": reasoning_chars,
                }

            async def consume_with_context_adjustment(
                args: dict[str, Any],
            ) -> tuple[str, dict[str, Any]]:
                current_args = args
                context_adjustments = 0
                connection_retries = 0
                capacity_retries = 0
                capacity_wait = 0.0
                while True:
                    try:
                        return await consume_stream(current_args)
                    except APIConnectionError as exc:
                        if connection_retries >= CONNECTION_RETRY_LIMIT:
                            raise
                        connection_retries += 1
                        await self.trace.emit(
                            "llm_connection_retry",
                            role=self.role,
                            stage=stage,
                            attempt=connection_retries,
                            error=str(exc),
                        )
                        await asyncio.sleep(CONNECTION_RETRY_DELAY_SECONDS)
                    except APIStatusError as exc:
                        if is_chain_capacity_error(exc):
                            if capacity_retries >= CHAIN_CAPACITY_RETRY_LIMIT:
                                raise
                            capacity_retries += 1
                            capacity_wait += CHAIN_CAPACITY_RETRY_DELAY_SECONDS
                            await self.trace.emit(
                                "chain_capacity_retry",
                                role=self.role,
                                stage=stage,
                                attempt=capacity_retries,
                                waited_seconds=capacity_wait,
                            )
                            await asyncio.sleep(CHAIN_CAPACITY_RETRY_DELAY_SECONDS)
                            continue
                        retry_max_tokens = context_retry_max_tokens(
                            exc, int(current_args["max_tokens"])
                        )
                        if retry_max_tokens is not None and context_adjustments < 4:
                            context_adjustments += 1
                            await self.trace.emit(
                                "context_window_output_adjusted",
                                role=self.role,
                                stage=stage,
                                requested_max_tokens=current_args["max_tokens"],
                                retry_max_tokens=retry_max_tokens,
                            )
                            current_args = dict(current_args)
                            current_args["max_tokens"] = retry_max_tokens
                            continue
                        sizes = context_window_sizes(exc)
                        compacted = (
                            compact_messages_for_context(
                                current_args["messages"],
                                prompt_tokens=sizes[0],
                                context_tokens=sizes[1],
                                requested_max_tokens=int(current_args["max_tokens"]),
                            )
                            if sizes is not None and context_adjustments < 4
                            else None
                        )
                        if compacted is None:
                            raise
                        context_adjustments += 1
                        compacted_messages, before_chars, after_chars = compacted
                        await self.trace.emit(
                            "context_window_input_adjusted",
                            role=self.role,
                            stage=stage,
                            reported_prompt_tokens=sizes[0],
                            context_tokens=sizes[1],
                            requested_max_tokens=current_args["max_tokens"],
                            before_chars=before_chars,
                            after_chars=after_chars,
                        )
                        current_args = dict(current_args)
                        current_args["messages"] = compacted_messages

            try:
                content, metadata = await consume_with_context_adjustment(request_args)
            except APIStatusError as exc:
                if not chain_id or not is_chain_rebuild_error(exc):
                    raise
                await self.trace.emit(
                    "chain_cache_recovery_started",
                    role=self.role,
                    stage=stage,
                    chain_id=chain_id,
                )
                retry_args = dict(request_args)
                retry_args["extra_body"] = {
                    key: value
                    for key, value in request_args["extra_body"].items()
                    if key not in {"chain_id", "chain_append_start"}
                }
                retry_args["messages"] = recovery_messages or request_args["messages"]
                content, metadata = await consume_with_context_adjustment(retry_args)
                await self.trace.emit(
                    "chain_cache_recovered",
                    role=self.role,
                    stage=stage,
                    evicted_chain_id=chain_id,
                    replacement_chain_id=metadata.get("chain_id"),
                )
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
        await self.trace.emit(
            "llm_request_completed",
            role=self.role,
            stage=stage,
            output_chars=len(content),
            **metadata,
        )
        return content, metadata

    async def structured(
        self,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        retries: int,
        max_tokens: int | None = None,
        salvage: Any = None,
    ) -> tuple[T, dict[str, Any]]:
        last_error: Exception | None = None
        request = user
        for attempt in range(retries + 1):
            text, metadata = await self.text(
                f"{stage}.attempt_{attempt + 1}", system, request, max_tokens=max_tokens
            )
            try:
                candidates = extract_json_candidates(text)
                validation_errors: list[ValidationError] = []
                for candidate in candidates:
                    try:
                        return schema.model_validate(candidate), metadata
                    except ValidationError as exc:
                        validation_errors.append(exc)
                # A truncated wrapper leaves only the inner objects decodable;
                # the caller can rebuild the whole result from those pieces.
                if salvage is not None:
                    rebuilt = salvage(candidates)
                    if rebuilt is not None:
                        result = schema.model_validate(rebuilt)
                        await self.trace.emit(
                            "structured_output_salvaged",
                            role=self.role,
                            stage=stage,
                            attempt=attempt + 1,
                            candidates=len(candidates),
                        )
                        return result, metadata
                raise validation_errors[0]
            except (ValueError, ValidationError) as exc:
                last_error = exc
                await self.trace.emit(
                    "structured_output_invalid",
                    role=self.role,
                    stage=stage,
                    attempt=attempt + 1,
                    error=str(exc),
                    output_preview=text[:1000],
                )
                required_fields = ", ".join(schema.model_fields)
                request = (
                    user
                    + "\n\nThe preceding response was not a valid result. Regenerate the "
                    + "complete result as exactly one JSON object. Do not return a JSON "
                    + "Schema, example, explanation, or Markdown fence. Ensure every array "
                    + "and object is closed. Required top-level fields: "
                    + f"{required_fields}."
                )
        raise RuntimeError(f"{stage} failed structured output validation: {last_error}")
