"""Application assembly and evaluator-compatible result export."""
from __future__ import annotations
from contextlib import asynccontextmanager
import asyncio
import json
import os
import time
from pathlib import Path
from .admission import PriorityGate
from .config import AppConfig
from .evidence import render_answer
from .retrieval import FixtureRetriever, McpRetriever, RecordedRetriever
from .runtime import Runtime
from .passage_runtime import PassageRuntime
from .storage import Store
from .tokenization import make_counter
from .transport import ChatClient, completion_url


async def observe_event_loop(store: Store, interval_seconds: float = 1.0):
    """Measure scheduling lateness without changing model or retriever deadlines."""
    while True:
        deadline = time.monotonic() + interval_seconds
        await asyncio.sleep(interval_seconds)
        observed = time.monotonic()
        store.event("event_loop_lag", lag_ms=max(0.0, observed - deadline) * 1000,
                    interval_seconds=interval_seconds)


@asynccontextmanager
async def open_runtime(config: AppConfig, store: Store):
    if selection := config.engine_manifest.get('configuration_selection'):
        store.event('configuration_selection', **selection)
    specs = {"main": config.main, "reader": config.reader}
    specs.update({f"reader_replica_{index}": replica
                  for index, replica in enumerate(config.reader_replicas, start=1)})
    if config.dense_reader:
        specs["dense_reader"] = config.dense_reader
    counters = {}
    for name, spec in specs.items():
        # Local HF initialization is done once per distinct tokenizer configuration.
        key = (spec.tokenizer_path, spec.tokenizer_revision, spec.tokenizer_format, spec.trust_remote_code, spec.engine, spec.enable_thinking,
               spec.preserve_thinking, json.dumps(spec.chat_template_kwargs, sort_keys=True))
        reuse = next((counters[n] for n, s in specs.items() if n in counters and
                      (s.tokenizer_path, s.tokenizer_revision, s.tokenizer_format, s.trust_remote_code, s.engine, s.enable_thinking,
                       s.preserve_thinking, json.dumps(s.chat_template_kwargs, sort_keys=True)) == key), None)
        counters[name] = reuse or make_counter(spec, config.workflow.allow_approximate_tokenizer)
    # Endpoints shared by Main/Reader also share one request gate, rather than each
    # independently claiming the full GPU concurrency allocation.
    capacities = {}
    for spec in specs.values():
        url = completion_url(spec.base_url)
        capacities[url] = min(capacities.get(url, spec.max_inflight), spec.max_inflight)
    gates = {url: PriorityGate(cap, config.workflow.reader_priority_aging_seconds)
             for url, cap in capacities.items()}
    clients = {n: ChatClient(s, store, gate=gates[completion_url(s.base_url)],
                            store_raw=config.workflow.store_raw_requests) for n, s in specs.items()}
    inner = None
    primary_error = None
    loop_observer = None
    try:
        if config.retrieval.transport == "fixture":
            inner = FixtureRetriever(json.loads(Path(config.retrieval.fixture_path).read_text(encoding="utf-8")))
        else:
            inner = await McpRetriever(config.retrieval, store=store).open()
        recorded = RecordedRetriever(inner, store, config.retrieval.cache_entries)
        runtime_class = PassageRuntime if config.workflow.research_protocol == "passages-v2" else Runtime
        runtime = runtime_class(config, store, recorded, clients, counters)
        if os.environ.get("BCGRAPH_OBSERVE_LOOP") == "1":
            loop_observer = asyncio.create_task(observe_event_loop(store), name="bcgraph-event-loop-observer")
        yield runtime
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors = []
        if loop_observer is not None:
            loop_observer.cancel()
            for result in await asyncio.gather(loop_observer, return_exceptions=True):
                if isinstance(result, Exception):
                    cleanup_errors.append(result)
        for client in clients.values():
            try:
                await client.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if inner is not None:
            try:
                await inner.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is None:
                raise ExceptionGroup('Runtime cleanup failed', cleanup_errors)
            try:
                store.event('runtime_cleanup_error', errors=[
                    f'{type(exc).__name__}: {exc}'[:600] for exc in cleanup_errors])
            except Exception:
                pass


def export_result(state: dict, config_hash: str, duration_ms: float, *,
                  store: Store | None = None, model: str | None = None) -> dict:
    cells = state.get("cells", {})
    all_usage = [m["usage"] for m in state.get("main_usage", [])]
    all_usage += [m["usage"] for c in cells.values() for m in c.get("metrics", [])]
    counts = {"search": sum(c.get("searches_used", 0) for c in cells.values()),
              "get_document": sum(c.get("documents_used", 0) for c in cells.values()),
              "reader": sum(c.get("turn", 0) for c in cells.values()),
              "main": len(state.get("main_usage", []))}
    docids = {d for c in cells.values() for d in c.get("doc_catalog", {})}
    operations = store.operations(state["scope"]) if store and state.get("scope") else []
    remedy = state.get("terminal_remedy", {})
    if store and state.get("scope"):
        committed = store.get("terminal-remedy:" + state["scope"])
        if committed:
            remedy = committed["value"]
    uncertain = sum(op["status"] in {"pending", "ambiguous"} for op in operations)
    if operations:
        # A failed child may never return its state; its committed work is still
        # in the journal. Scope excludes previous attempts and other queries.
        counts = dict.fromkeys(counts, 0)
        all_usage = []
        for op in operations:
            key = op["key"]
            if key.startswith("tool:"):
                kind = "search" if ":search:" in key else "get_document"
                counts[kind] += 1
                if kind == "search" and op["status"] == "success":
                    docids.update(str(hit["docid"]) for hit in op["value"])
            else:
                counts["reader" if ":reader:" in key else "main"] += 1
                if op["status"] == "success":
                    all_usage.append(op["value"]["usage"])
    def total(key):
        values = [u.get(key) for u in all_usage]
        return sum(values) if not uncertain and all(v is not None for v in values) else None
    errors = [str(e) for e in state.get("errors", [])]
    bad_final = any(e.startswith("final_format_error:") for e in errors)
    service_error = any(c.get("last_error") for c in cells.values()) or any(
        "Main service error:" in e or "Main synchronization service error:" in e for e in errors)
    final_format_failures = max(state.get("final_format_failures", 0), int(bad_final),
                               int(remedy.get("kind") == "output_recovery"))
    format_failures = sum(c.get("reader_format_failures", 0) for c in cells.values()) + final_format_failures
    partial_recoveries = sum(c.get("reader_partial_recoveries", 0) for c in cells.values())
    planner_fallbacks = sum(e.startswith("planner_format_fallback:") for e in errors)
    final_rejections = sum(e.startswith("final_validation_failed:") for e in errors)
    legacy = state.get("status", "error")
    execution = ("error" if legacy not in {"completed", "unresolved"} or bad_final or uncertain
                 else "degraded" if format_failures or service_error or partial_recoveries or planner_fallbacks else "ok")
    outcome = ("answered" if legacy == "completed" else "failed" if execution == "error"
               else "abstained" if legacy == "unresolved" else "failed")
    if state.get("protocol") == "passages-v2" and execution == "ok" and state.get("final_repair_count", 0):
        execution = "degraded"
    if execution == 'ok' and any(e.startswith('coordination_error:') for e in errors):
        execution = 'degraded'
    return {"query_id": str(state["query_id"]), "status": legacy,
            "tool_call_counts": counts,
            "retrieved_docids": sorted(docids),
            "result": [{"type": "output_text", "output": state.get("final_text") or render_answer(state.get("decision", {}))}],
            "usage": {"reported_prompt_tokens": total("reported_prompt_tokens"),
                      "output_tokens": total("completion_tokens"), "cache_read_tokens": total("reused_tokens"),
                      "prefilled_tokens": total("prefilled_tokens")},
            "metadata": {"attempt": state.get("attempt", 1), "scope": state.get("scope"),
                         "model": model, "harness": "persistent-cell-langgraph",
                         "counts_source": "operation_journal" if operations else "graph_state",
                         "uncertain_operations": uncertain,
                         "duration_ms": duration_ms, "answer_support": state.get("answer_support", "none"),
                         "cell_stop_reasons": {i: c.get("stop_reason") for i, c in cells.items()},
                         "reopens": state.get("reopens", 0), "errors": state.get("errors", []),
                         "execution_status": execution, "outcome": outcome,
                         "protocol_failure_count": format_failures,
                         "reader_delivery_repairs": sum(c.get('reader_delivery_repairs', 0) for c in cells.values()),
                         "final_format_failure_count": final_format_failures,
                         "terminal_remedy": remedy,
                         "query_deadline": state.get("query_deadline"),
                         "planner_fallback_count": planner_fallbacks,
                         "final_validation_rejection_count": final_rejections,
                         "context_citation_count": len(state.get("decision", {}).get("context_evidence_ids", [])),
                         "reader_partial_recoveries": partial_recoveries,
                         "shown_docids": sorted({s["docid"] for c in cells.values() for s in c.get("sources", {}).values()}),
                         "accepted_evidence_count": sum(sum(e.get("active", True) for e in c.get("evidence", {}).values()) for c in cells.values()),
                         "research_protocol": state.get("protocol", "legacy-evidence"),
                         "selected_passage_count": sum(sum(p.get("active", True) for p in c.get("selected_passages", {}).values()) for c in cells.values()),
                         "final_repair_count": state.get("final_repair_count", 0),
                         "final_review_count": state.get("final_review_count", 0),
                         "delivery_issues": state.get("delivery_issues", []),
                         "decision_exit_cause": state.get("decision_exit_cause"),
                         "semantic_correctness_verified": False,
                         "confidence_evaluated": False,
                         "accuracy_evaluated": False,
                         "note": "Per-result usage counts committed logical requests once. _meta/events.jsonl also includes retries and unknown work."},
            "config_hash": config_hash}
