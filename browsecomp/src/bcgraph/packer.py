"""Relevant contiguous source windows, full-chat token bounds, and source versioning."""
from __future__ import annotations
import re
from typing import Any
from .config import WorkflowConfig
from .evidence import source_record
from .prompts import reader_message
from .tokenization import TokenCounter, prefix_chars


def _window_start(document: dict, query: str, previous: dict, preferred: int | None) -> int | None:
    text, docid = document["text"], document["docid"]
    if preferred is not None:
        return preferred if preferred < len(text) else None
    seen = sorted((s["start"], s["end"]) for s in previous.values() if s["docid"] == docid)
    # Distinct current-round terms come first; repeated broad keywords no longer
    # dominate every paragraph solely because the question/focus repeats them.
    stopwords = {"article", "person", "individual", "following", "according", "which", "their", "between", "about", "provide"}
    terms = list(dict.fromkeys(t.lower() for t in re.findall(r"\w+", query)
                              if len(t) >= 4 and t.lower() not in stopwords))[:60]
    points = {0}
    for _, end in seen:
        if end < len(text):
            points.add(end)
    # Contiguous windows remain citable: no concatenation of unrelated fragments.
    for match in re.finditer(r"\n\s*\n", text):
        points.add(match.end())
    candidates = []
    for start in sorted(points):
        if any(a <= start < b for a, b in seen):
            continue
        snippet = text[start:start + 3500].lower()
        score = sum(snippet.count(t) for t in terms)
        candidates.append((-score, start))
    return min(candidates)[1] if candidates else None


def pack_documents(state: dict, documents: list[dict], counter: TokenCounter,
                   cfg: WorkflowConfig, max_context: int, max_output: int) -> dict[str, Any]:
    first = state["turn"] == 0
    source_budget = cfg.first_source_tokens if first else cfg.followup_source_tokens
    repair_reserve = cfg.reader_delivery_repair_tokens + cfg.future_turn_overhead_tokens if cfg.reader_delivery_repair_tokens else 0
    hard_cap = max_context - max_output - cfg.context_reserve_tokens - repair_reserve
    cap = hard_cap
    history = state["raw_history"]
    sources: dict[str, dict] = {}
    empty_tokens = counter.messages([*history, reader_message(state, {})])
    # Do not use almost the entire context in the initial rounds and then
    # advertise a reopen that physically cannot fit. Reserve future increments,
    # outputs and bounded message overhead; retain an authoritative hard check.
    reserved_future = 0
    if state.get("protocol") == "passages-v2":
        future_rounds = max(0, state["turn_limit"] - state["turn"] - 1)
        if state.get("revision", 1) == 1:
            future_rounds += cfg.max_reopens * cfg.reopen_turns
        per_round = cfg.followup_source_tokens + cfg.reader_followup_output_tokens + cfg.future_turn_overhead_tokens + repair_reserve
        reserved_future = future_rounds * per_round
        cap = min(hard_cap, max(empty_tokens + cfg.min_source_tokens + 256, hard_cap - reserved_future))
    if empty_tokens >= hard_cap:
        return {"message": None, "sources": {}, "logical_prompt_tokens": empty_tokens,
                "stop_reason": "context_budget_exhausted"}
    used = 0
    for index, doc in enumerate(documents):
        if used >= source_budget:
            break
        remaining = source_budget - used
        fair_share = max(cfg.min_source_tokens, remaining // max(1, len(documents) - index))
        limit = min(remaining, cfg.max_source_tokens_per_document, fair_share)
        start = _window_start(doc, state.get("retrieval_context", "") + " " + state["question"] + " " + state["focus"],
                              state["sources"], doc.get("requested_offset"))
        if start is None:
            continue
        length = prefix_chars(doc["text"][start:], limit, counter)
        if not length:
            continue
        while length > 0:
            source = source_record(doc["docid"], doc["text"], start, start + length, doc.get("title", ""))
            trial = {**sources, source["source_id"]: source}
            message = reader_message(state, trial)
            if counter.messages([*history, message]) <= cap:
                break
            length = int(length * 0.8)
        if length <= 0:
            continue
        if source["source_id"] in state["sources"] and doc.get("requested_offset") is None:
            continue
        # Small but useful genuine documents may be shorter than min_source_tokens.
        sources[source["source_id"]] = source
        used += counter.text(source["text"])
    if not sources:
        return {"message": None, "sources": {}, "logical_prompt_tokens": empty_tokens,
                "stop_reason": "no_novel_evidence"}
    message = reader_message(state, sources)
    count = counter.messages([*history, message])
    if count > cap:
        raise AssertionError("Packer exceeded full logical context cap")
    return {"message": message, "sources": sources, "source_tokens": used,
            "logical_prompt_tokens": count, "token_count_exact": counter.exact, "stop_reason": "", "hard_prompt_cap": hard_cap, "scheduled_prompt_cap": cap, "reserved_future_tokens": reserved_future}
