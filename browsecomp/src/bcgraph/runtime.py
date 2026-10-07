"""Node implementations shared by the actual LangGraph and independent logic tests."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import time
from typing import Any

from .admission import PriorityGate
from .config import AppConfig
from .delivery import partition_decision_citations
from .evidence import (accept_reply, candidate_status, flatten_evidence, merge_sources,
                       normalize_query, render_answer, stable_hash, validate_final, evidence_aliases)
from .packer import pack_documents
from .prompts import MAIN_SYSTEM, MAIN_THINKING_SYSTEM, READER_SYSTEM, final_message, planner_message, reader_message, js
from .retrieval import RecordedRetriever, RetrievalError
from .schemas import (FinalDecision, Plan, ReaderReply, message_text, parse_json_object, parse_reader_output, parse_plan_output)
from .storage import Store
from .tokenization import TokenCounter
from .transport import ChatClient, ModelRequestError


def unique_queries(values: list[str], seen: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    used = set(seen)
    out = []
    for raw in values:
        query = " ".join(raw.split())[:600]
        key = normalize_query(query)
        if key and key not in used:
            used.add(key)
            out.append(query)
        if len(out) >= limit:
            break
    return out


def distribute(total: int, count: int) -> list[int]:
    return [total // count + int(i < total % count) for i in range(count)]


def fallback_plan(question: str) -> dict:
    return {"constraints": [{"id": "target", "description": question, "time_scope": "unspecified",
                             "required": True, "answer_target": True}],
            "cells": [{"id": "cell1", "focus": question, "constraint_ids": ["target"],
                       "initial_queries": [question]}]}


class Runtime:
    def __init__(self, config: AppConfig, store: Store, retrieval: RecordedRetriever,
                 clients: dict[str, ChatClient], counters: dict[str, TokenCounter]):
        self.config, self.store, self.retrieval = config, store, retrieval
        self.clients, self.counters = clients, counters
        self.reader_backends = tuple(
            name for name in clients
            if name == "reader" or name.startswith("reader_replica_")
        )
        self._reader_replica_cursor = 0
        self.cost_router = None
        if config.workflow.reader_routing == 'measured':
            from .reader_costs import BenefitRouter
            self.cost_router = BenefitRouter(config, clients, counters)
        self.live_cells = PriorityGate(config.workflow.max_live_cells,
                                       config.workflow.reader_priority_aging_seconds)

    def assign_reader_backend(self, state: dict) -> str:
        if state.get("backend"):
            return state["backend"]
        backend = self.reader_backends[self._reader_replica_cursor % len(self.reader_backends)]
        self._reader_replica_cursor += 1
        return backend

    async def plan(self, state: dict) -> dict:
        cfg = self.config.workflow
        system = MAIN_THINKING_SYSTEM if self.config.main.enable_thinking is True else MAIN_SYSTEM
        history = [{"role": "system", "content": system},
                   planner_message(state["question"], cfg.max_cells_per_query)]
        count = self.counters["main"].messages(history)
        if count + cfg.planner_output_tokens + cfg.context_reserve_tokens > self.config.main.max_context_tokens:
            raise ModelRequestError("Question/plan prompt exceeds Main context budget")
        reply = await self.clients["main"].complete(
            history, cfg.planner_output_tokens, operation_id=state["scope"] + ":main:plan",
            writer_key=state["scope"] + ":main", local_prompt_tokens=count)
        errors = []
        try:
            if reply["assistant"].get("tool_calls"):
                raise ValueError("Planner produced unexpected tool calls")
            parsed_plan, plan_notes = parse_plan_output(message_text(reply["assistant"]), max_cells=cfg.max_cells_per_query)
            plan = parsed_plan.model_dump()
            errors.extend("planner_normalized: " + note for note in plan_notes)
        except ValueError as exc:
            plan = fallback_plan(state["question"])
            errors.append("planner_format_fallback: " + str(exc)[:800])
        if not any(c["required"] and not c["answer_target"] for c in plan["constraints"]):
            errors.append("planner_target_only: automatic evidence-sufficient stopping disabled")
        cells = plan["cells"]
        if len(cells) > cfg.max_cells_per_query:
            # Preserve all assigned objectives rather than dropping excess cells.
            cells = [{"id": "cell1", "focus": "\n".join(c["focus"] for c in cells),
                      "constraint_ids": list(dict.fromkeys(i for c in cells for i in c["constraint_ids"])),
                      "initial_queries": unique_queries([q for c in cells for q in c["initial_queries"]], [], 6)}]
            errors.append("planner_cells_merged_to_configured_limit")
        n = len(cells)
        reserve_searches = min(cfg.reopen_searches, max(0, cfg.max_searches_per_query - n)) if cfg.max_reopens else 0
        reserve_docs = min(cfg.followup_documents * cfg.reopen_turns, max(0, cfg.max_document_fetches_per_query - n)) if cfg.max_reopens else 0
        reserve_output = min(cfg.reader_followup_output_tokens * cfg.reopen_turns,
                             max(0, cfg.max_total_reader_output_tokens - n * 128)) if cfg.max_reopens else 0
        searches = distribute(cfg.max_searches_per_query - reserve_searches, n)
        docs = distribute(cfg.max_document_fetches_per_query - reserve_docs, n)
        outputs = distribute(cfg.max_total_reader_output_tokens - reserve_output, n)
        jobs = []
        for i, cell in enumerate(cells):
            jobs.append({**cell, "search_limit": searches[i], "document_limit": docs[i],
                         "output_limit": outputs[i], "turn_limit": cfg.max_reader_turns})
        self.store.event("plan", scope=state["scope"], query_id=state["query_id"], cells=n,
                         errors=errors, constraints=plan["constraints"])
        return {"constraints": {c["id"]: c for c in plan["constraints"]}, "jobs": jobs,
                "main_history": [*history, reply["assistant"]], "cells": {},
                "decision_round": 0, "reopens": 0, "errors": errors,
                "main_usage": [{"stage": "plan", "usage": reply["usage"]}]}

    def init_cell(self, worker: dict) -> dict:
        job = worker["job"]
        old = worker.get("previous")
        if old:
            state = deepcopy(old)
            state.update(revision=old["revision"] + 1, stop_reason="", last_error="",
                         pending_queries=job.get("initial_queries", []),
                         pending_read_more=job.get("read_more", []),
                         search_limit=job["search_limit"], document_limit=job["document_limit"],
                         output_limit=job["output_limit"], turn_limit=job["turn_limit"],
                         no_progress_turns=0, reopen_instruction=job.get("instruction", ""),
                         packed={}, reply={})
            return state
        return {"query_id": worker["query_id"], "question": worker["question"], "scope": worker["scope"],
                "cell_id": job["id"], "revision": 1, "focus": job["focus"],
                "constraint_ids": job["constraint_ids"], "constraints": worker["constraints"],
                "search_limit": job["search_limit"], "document_limit": job["document_limit"],
                "output_limit": job["output_limit"], "turn_limit": job["turn_limit"],
                "pending_queries": job["initial_queries"], "pending_read_more": [],
                "raw_history": [{"role": "system", "content": READER_SYSTEM}],
                "handle": None, "backend": "", "sources": {}, "evidence": {},
                "doc_catalog": {}, "seen_queries": [], "searches_used": 0, "documents_used": 0,
                "output_used": 0, "turn": 0, "no_progress_turns": 0, "stop_reason": "",
                "summary": "", "validation_errors": [], "retrieval_notes": [], "metrics": [],
                "last_error": "", "packed": {}, "reply": {}, "reopen_instruction": "",
                "reader_format_failures": 0, "reader_partial_recoveries": 0, "protocol_issues": [],
                "retrieval_context": ""}

    def _fallback_queries(self, state: dict) -> list[str]:
        statuses = candidate_status(state["evidence"], state["constraints"])
        if statuses:
            best = max(statuses.values(), key=lambda r: (len(r["supported"]), len(r["answers"])))
            gaps = best["missing"] or [i for i in state["constraint_ids"] if i in best["contradicted"]]
            return [best["candidate"] + " " + state["constraints"][i]["description"] for i in gaps[:2]]
        return [state["constraints"][i]["description"] for i in state["constraint_ids"][:2]]

    async def cell_fetch(self, state: dict) -> dict:
        cfg = self.config.workflow
        if not state.get("backend") and len(self.reader_backends) > 1:
            state = {**state, "backend": self.assign_reader_backend(state)}
        feedback_pending = bool(state.get('reopen_instruction')) and (
            state.get('last_consumed_feedback_revision', 0) < state['revision'])
        if state["turn"] >= state["turn_limit"]:
            return {"stop_reason": "turn_budget_exhausted", "packed": {"message": None}}
        if state["output_limit"] - state["output_used"] < 128:
            return {"stop_reason": "output_budget_exhausted", "packed": {"message": None}}
        if state["documents_used"] >= state["document_limit"] and not feedback_pending:
            return {"stop_reason": "document_budget_exhausted", "packed": {"message": None}}
        remaining_search = max(0, state["search_limit"] - state["searches_used"])
        pending = unique_queries(state["pending_queries"], state["seen_queries"], 20)
        read_more = state.get("pending_read_more", [])
        if not pending and not read_more and remaining_search:
            pending = unique_queries(self._fallback_queries(state), state["seen_queries"], 2)
        take = min(remaining_search, cfg.max_searches_per_fetch)
        queries, pending_rest = pending[:take], pending[take:]
        if state['documents_used'] >= state['document_limit']:
            queries, pending_rest, read_more = [], pending, []
        catalog = deepcopy(state["doc_catalog"])
        notes: list[str] = []
        results = await asyncio.gather(*[
            self.retrieval.search(q, f'{state["scope"]}:cell:{state["cell_id"]}:search:{state["searches_used"] + i + 1}')
            for i, q in enumerate(queries)], return_exceptions=True)
        hits_by_query = []
        for query, result in zip(queries, results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if not isinstance(result, RetrievalError):
                    raise result
                notes.append(f"Search failed for {query}: {result}")
                hits_by_query.append([])
            else:
                hits_by_query.append(result)
                for hit in result:
                    catalog.setdefault(hit["docid"], hit)
        # Interleave ranked results rather than taking all documents from search 1.
        hits = []
        for rank in range(max((len(r) for r in hits_by_query), default=0)):
            for result in hits_by_query:
                if rank < len(result):
                    hits.append(result[rank])
        if state.get("protocol") == "passages-v2":
            hit_ids = {h["docid"] for h in hits}
            hits.extend(v for k, v in catalog.items() if k not in hit_ids)
        document_cap = cfg.first_documents if state["turn"] == 0 else cfg.followup_documents
        document_cap = min(document_cap, state["document_limit"] - state["documents_used"])
        requests, keys = [], set()
        known = set(catalog) | {s["docid"] for s in state["sources"].values()}
        for req in read_more:
            key = (req["docid"], req["offset"])
            if req["docid"] not in known:
                notes.append(f"Rejected read_more for undiscovered docid {req['docid']}")
            elif key not in keys:
                requests.append({"docid": req["docid"], "requested_offset": req["offset"]})
                keys.add(key)
        for hit in hits:
            docid = hit["docid"]
            if any(r["docid"] == docid for r in requests):
                continue
            prior = [s for s in state["sources"].values() if s["docid"] == docid]
            if any(s["start"] == 0 and s["end"] == s["total_chars"] for s in prior):
                continue
            requests.append({"docid": docid})
        selection_updates = {}
        if cfg.researcher_select_documents and not read_more and document_cap > 0:
            requests, selection_updates = await self.select_documents(
                {**state, 'doc_catalog': catalog, 'retrieval_context': ' '.join(queries)},
                requests, document_cap)
            if selection_updates.get('stop_reason'):
                return {**selection_updates, 'doc_catalog': catalog,
                        'searches_used': state['searches_used'] + len(queries),
                        'seen_queries': [*state['seen_queries'], *map(normalize_query, queries)],
                        'packed': {'message': None}}
        else:
            requests = requests[:max(0, document_cap)]
        fetched = await asyncio.gather(*[
            self.retrieval.get_document(r["docid"],
                f'{state["scope"]}:cell:{state["cell_id"]}:document:{state["documents_used"] + i + 1}')
            for i, r in enumerate(requests)], return_exceptions=True)
        documents = []
        for req, result in zip(requests, fetched):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if not isinstance(result, RetrievalError):
                    raise result
                notes.append(str(result))
            elif result is None:
                notes.append(f"Document not found: {req['docid']}")
            elif result["text"]:
                documents.append({**result, **req})
        updates = {**selection_updates, "doc_catalog": catalog, "seen_queries": [*state["seen_queries"], *map(normalize_query, queries)],
                   "searches_used": state["searches_used"] + len(queries),
                   "documents_used": state["documents_used"] + len(requests),
                   "pending_queries": pending_rest, "pending_read_more": [],
                   "retrieval_notes": [*state["retrieval_notes"], *notes]}
        # The next passage should follow THIS round's question, not only the
        # initial broad identity query. This does not add a model call.
        updates["retrieval_context"] = " ".join(queries)
        if selection_updates.get('selection_next_queries'):
            updates['pending_queries'] = unique_queries(
                [*selection_updates['selection_next_queries'], *pending_rest],
                updates['seen_queries'], 8)
        view = {**state, **updates}
        max_output = min(cfg.reader_first_output_tokens if state["turn"] == 0 else cfg.reader_followup_output_tokens,
                         state["output_limit"] - view["output_used"])
        backend = state["backend"] or "reader"
        endpoint = self.clients[backend].config
        packed = await self._pack_documents(view, documents, backend, max_output)
        feedback_pending = bool(state.get('reopen_instruction')) and (
            state.get('last_consumed_feedback_revision', 0) < state['revision'])
        if cfg.researcher_select_documents and not packed['message'] and feedback_pending:
            message = reader_message(view, {})
            count = self.counters[backend].messages([*view['raw_history'], message])
            if count + max_output + cfg.context_reserve_tokens < endpoint.max_context_tokens:
                packed = {'message': message, 'sources': {}, 'source_tokens': 0,
                          'logical_prompt_tokens': count, 'stop_reason': ''}
        updates['fetch_route'] = 'read'
        if cfg.researcher_select_documents and not packed['message']:
            fresh = unique_queries(updates['pending_queries'], updates['seen_queries'], 8)
            can_acquire = bool(fresh) and updates['searches_used'] < state['search_limit']
            updates['fetch_route'] = 'acquire' if can_acquire else 'yield'
            packed['stop_reason'] = '' if can_acquire else 'no_new_lead'
            self.store.event('read_skipped', scope=state['scope'], cell_id=state['cell_id'],
                             turn=state['turn'], reason=updates['fetch_route'])
        if (not state["backend"] and cfg.cold_dense_below_tokens and packed["message"]
                and packed["logical_prompt_tokens"] < cfg.cold_dense_below_tokens):
            backend = "dense_reader"
            endpoint = self.clients[backend].config
            packed = await self._pack_documents(view, documents, backend, max_output)
        if not packed["message"] and notes and not state["sources"] and updates.get("fetch_route") != "acquire":
            packed["stop_reason"] = "retrieval_failed"
        updates.update(packed=packed, backend=backend, requested_output_tokens=max_output,
                       stop_reason=packed["stop_reason"])
        self.store.event("cell_pack", scope=state["scope"], cell_id=state["cell_id"],
                         turn=state["turn"] + 1, backend=backend,
                         source_count=len(packed["sources"]), source_tokens=packed.get("source_tokens", 0),
                         logical_prompt_tokens=packed["logical_prompt_tokens"],
                         token_count_exact=self.counters[backend].exact, stop_reason=packed["stop_reason"])
        return updates

    async def _pack_documents(self, state: dict, documents: list[dict],
                              backend: str, max_output: int) -> dict:
        def pack():
            # Tokenizers can mutate encoding settings. Isolate the worker's copy
            # while retaining the exact source selection and token accounting.
            counter = deepcopy(self.counters[backend])
            return pack_documents(state, documents, counter, self.config.workflow,
                                  self.clients[backend].config.max_context_tokens, max_output)

        return await asyncio.to_thread(pack)

    def reader_route(self, state: dict) -> tuple[str, str]:
        cfg = self.config.workflow
        if cfg.reader_routing == "off":
            return state["backend"], "configured"
        handle = state.get("handle")
        client = self.clients["reader"]
        if (cfg.reader_routing == "balanced" and client.config.method == "vanilla"
                and state["turn"] > 0 and state.get("backend")):
            return state["backend"], "retained_dense_prefix"
        if (handle and handle.get("chain_id") and handle.get("endpoint") == client.url
                and handle.get("engine_epoch") == client.config.engine_epoch
                and handle.get("client_epoch") == client.client_epoch):
            return "reader", "retained_chain"
        loads = {name: self.clients[name].gate.snapshot() for name in ("reader", "dense_reader")}
        if (cfg.reader_routing == "comfort"
                and state["packed"]["logical_prompt_tokens"] < cfg.routing_h2o_min_prompt_tokens):
            return "dense_reader", "short_prompt"
        pressure = {name: (load["active"] + load["pending"]) / load["capacity"]
                    for name, load in loads.items()}
        backend = "reader" if pressure["reader"] < pressure["dense_reader"] else "dense_reader"
        return backend, "eligible_lower_queue" if cfg.reader_routing == "comfort" else "balanced_queue"

    async def cell_read(self, state: dict) -> dict:
        packed = state["packed"]
        messages = [*state["raw_history"], packed["message"]]
        route_detail = None
        backend, route_reason = state['backend'], 'configured'
        if self.config.workflow.reader_routing != "off":
            route_key = f'route:{state["scope"]}:reader:{state["cell_id"]}:turn:{state["turn"] + 1}'
            route_hash = stable_hash({"messages": messages, "config": self.config.model_dump()})
            saved = self.store.get(route_key, route_hash)
            if saved:
                backend, route_reason = saved["value"]["backend"], saved["value"]["reason"]
                route_detail = saved['value'].get('detail')
            else:
                if self.cost_router is not None:
                    backend, route_detail = await self.cost_router.choose(state)
                    route_reason = route_detail['reason']
                else:
                    backend, route_reason = self.reader_route(state)
                self.store.put(route_key, route_hash, "success", {
                    "backend": backend, "reason": route_reason, 'detail': route_detail})
            self.store.event("reader_route", scope=state["scope"], cell_id=state["cell_id"],
                             turn=state["turn"] + 1, backend=backend, reason=route_reason,
                             logical_prompt_tokens=packed["logical_prompt_tokens"],
                             cost_decision=route_detail,
                             loads={n: self.clients[n].gate.snapshot() for n in ("reader", "dense_reader")})
        try:
            reply = await self.clients[backend].complete(
                messages, state["requested_output_tokens"],
                operation_id=f'{state["scope"]}:reader:{state["cell_id"]}:turn:{state["turn"] + 1}',
                writer_key=f'{state["scope"]}:reader:{state["cell_id"]}', handle=state["handle"],
                continuation=state["turn"] > 0, local_prompt_tokens=packed["logical_prompt_tokens"])
        except ModelRequestError as exc:
            return {"stop_reason": "reader_service_error", "last_error": str(exc),
                    "output_used": state["output_used"] + state["requested_output_tokens"]}
        finally:
            if self.cost_router is not None:
                self.cost_router.release(state)
        actual_output = reply["usage"]["completion_tokens"]
        charge = actual_output if actual_output is not None else state["requested_output_tokens"]
        metric = {k: reply[k] for k in ("request_mode", "finish_reason", "duration_ms", "usage", "recoveries", "journal_replay")}
        metric.update({k: reply.get(k) for k in ("http_ms", "admission_wait_ms", "admission_snapshot")})
        metric.update(turn=state["turn"] + 1, output_budget_charged=charge,
                      backend=backend, route_reason=route_reason,
                      local_logical_prompt_tokens=packed["logical_prompt_tokens"])
        small_reply = {k: reply[k] for k in ("assistant", "usage", "finish_reason")}
        self.store.event('read_executed', scope=state['scope'], cell_id=state['cell_id'],
                         turn=state['turn'] + 1, feedback_revision=state.get('revision', 0),
                         source_count=len(packed['sources']),
                         selection_operation_id=state.get('selection_operation_id'),
                         source_docids=sorted({s['docid'] for s in packed['sources'].values()}))
        return {"last_consumed_feedback_revision": state.get('revision', 0),
                "backend": backend, "raw_history": [*messages, deepcopy(reply["assistant"])],
                "sources": merge_sources(state["sources"], packed["sources"]),
                "handle": reply["handle"], "turn": state["turn"] + 1,
                "output_used": state["output_used"] + charge, "reply": small_reply,
                "metrics": [*state["metrics"], metric]}

    def cell_validate(self, state: dict) -> dict:
        cfg = self.config.workflow
        errors, changes = [], 0
        partial, format_failed = False, False
        reader_requested_followup = False
        ledger = state["evidence"]
        try:
            assistant = state["reply"]["assistant"]
            if assistant.get("tool_calls"):
                return {"stop_reason": "unexpected_tool_calls", "last_error": "Reader has no tool-call protocol"}
            parsed, partial = parse_reader_output(message_text(assistant), state["reply"].get("finish_reason"))
            ledger, errors, changes = accept_reply(parsed, state["sources"], state["constraints"],
                                                   state["evidence"], state["cell_id"], state["turn"])
            queries = unique_queries([*parsed.next_queries, *state["pending_queries"]], state["seen_queries"], 8)
            read_more = [r.model_dump() for r in parsed.read_more]
            summary = parsed.summary or state["summary"]
            # Coverage labels are not a semantic proof. Do not mark a cell
            # sufficient while its own Reader requests executable verification.
            # Only explicit fresh leads count, not leftover planner queries.
            fresh_queries = unique_queries(parsed.next_queries, state["seen_queries"], 6)
            known_docs = set(state.get("doc_catalog", {})) | {
                source["docid"] for source in state["sources"].values()}
            usable_read_more = any(r["docid"] in known_docs for r in read_more)
            reader_requested_followup = bool(
                (fresh_queries and state["searches_used"] < state["search_limit"])
                or usable_read_more)
            if partial:
                errors.append("reader_partial_output: accepted only fully decoded and source-validated entries; no completion inferred")
        except ValueError as exc:
            format_failed = True
            errors = ["reader_format_error: " + str(exc)[:1000]]
            queries = unique_queries([*state["pending_queries"], *self._fallback_queries(state)], state["seen_queries"], 3)
            read_more, summary = [], state["summary"]
        no_progress = 0 if changes else state["no_progress_turns"] + 1
        assigned = {i: state["constraints"][i] for i in state["constraint_ids"]}
        statuses = candidate_status(ledger, assigned)
        needs_answer = any(c["answer_target"] for c in assigned.values())
        sufficient = [r for r in statuses.values() if not r["missing"] and not r["has_required_contradiction"]
                      and (not needs_answer or len(r["answers"]) == 1)]
        # One bounded grace round for a novel search/read lead, not an
        # unlimited exemption from no-progress stopping. The next unproductive
        # round (threshold+1) stops; turn/doc/output caps always take precedence.
        known = set(state.get("doc_catalog", {})) | {x["docid"] for x in state["sources"].values()}
        executable_lead = bool(
            (state["searches_used"] < state["search_limit"]
             and unique_queries(queries, state["seen_queries"], 1))
            or any(r["docid"] in known for r in read_more))
        progress_grace = bool(cfg.answer_policy == "best_effort" and not format_failed
                              and no_progress == cfg.max_no_progress_turns
                              and executable_lead)
        stop = ""
        has_identity_check = any(c.get("required", True) and not c.get("answer_target") for c in state["constraints"].values())
        if (len(sufficient) == 1 and has_identity_check and not partial and not format_failed
                and not reader_requested_followup):
            stop = "evidence_sufficient"
        elif state["turn"] >= state["turn_limit"]:
            stop = "turn_budget_exhausted"
        elif state["output_limit"] - state["output_used"] < 128:
            stop = "output_budget_exhausted"
        elif no_progress >= cfg.max_no_progress_turns and not progress_grace:
            stop = "no_evidence_progress"
        elif state["documents_used"] >= state["document_limit"]:
            stop = "document_budget_exhausted"
        elif not queries and not read_more:
            queries = unique_queries(self._fallback_queries({**state, "evidence": ledger}), state["seen_queries"], 2)
            if not queries or state["searches_used"] >= state["search_limit"]:
                stop = "no_new_lead"
        self.store.event("cell_validated", scope=state["scope"], cell_id=state["cell_id"], turn=state["turn"],
                         accepted_changes=changes, validation_errors=errors, stop_reason=stop,
                         partial_recovery=partial, format_failed=format_failed,
                         no_progress_grace_used=bool(progress_grace and not stop),
                         finish_reason=state["reply"].get("finish_reason"),
                         evidence_sufficient_deferred=bool(len(sufficient) == 1 and reader_requested_followup))
        return {"evidence": ledger, "validation_errors": errors, "pending_queries": queries,
                "pending_read_more": read_more, "summary": summary,
                "no_progress_turns": no_progress, "stop_reason": stop,
                "reader_format_failures": state.get("reader_format_failures", 0) + int(format_failed),
                "reader_partial_recoveries": state.get("reader_partial_recoveries", 0) + int(partial),
                "protocol_issues": [*state.get("protocol_issues", []), *errors]}

    def collect(self, state: dict) -> dict:
        evidence = flatten_evidence(state["cells"])
        return {"evidence": evidence, "candidates": candidate_status(evidence, state["constraints"])}

    def _reopen_job(self, state: dict, decision: FinalDecision) -> dict | None:
        cfg = self.config.workflow
        if state["reopens"] >= cfg.max_reopens or not decision.reopen_cell_id:
            return None
        cell = state["cells"].get(decision.reopen_cell_id)
        if cell is None or cell.get("last_error"):
            return None
        used_searches = sum(c["searches_used"] for c in state["cells"].values())
        used_docs = sum(c["documents_used"] for c in state["cells"].values())
        used_output = sum(c["output_used"] for c in state["cells"].values())
        searches = min(cfg.reopen_searches, cfg.max_searches_per_query - used_searches)
        docs = min(cfg.followup_documents * cfg.reopen_turns, cfg.max_document_fetches_per_query - used_docs)
        output = min((cfg.reader_followup_output_tokens + cfg.reader_delivery_repair_tokens) * cfg.reopen_turns,
                     cfg.max_total_reader_output_tokens - used_output)
        queries = unique_queries(decision.next_queries, cell["seen_queries"], max(0, searches)) if searches > 0 else []
        known = set(cell.get("doc_catalog", {})) | {s["docid"] for s in cell.get("sources", {}).values()}
        read_more = []
        for request in decision.read_more:
            r = request.model_dump()
            if r["docid"] not in known or r in read_more:
                continue
            lengths = [s["total_chars"] for s in cell.get("sources", {}).values()
                       if s["docid"] == r["docid"] and s.get("total_chars") is not None]
            if lengths and r["offset"] >= max(lengths):
                continue
            read_more.append(r)
        read_more = read_more[:max(0, docs)]
        if docs < 1 or output < 128 or (not queries and not read_more):
            return None
        if not self._has_reader_context(cell):
            return None
        return {"id": cell["cell_id"], "focus": cell["focus"], "constraint_ids": cell["constraint_ids"],
                "initial_queries": queries, "read_more": read_more,
                "search_limit": cell["searches_used"] + max(0, searches),
                "document_limit": cell["documents_used"] + docs,
                "output_limit": cell["output_used"] + output,
                "turn_limit": cell["turn"] + cfg.reopen_turns, "instruction": decision.explanation}

    def _has_reader_context(self, cell: dict) -> bool:
        """Sparse physical KV does not increase the configured logical context."""
        if not cell.get("raw_history"):
            return True  # no cached history yet; normal packer handles the first call
        backend = cell.get("backend") or "reader"
        cfg = self.config.workflow
        count = self.counters[backend].messages([*cell["raw_history"], reader_message(cell, {})])
        output = min(cfg.reader_followup_output_tokens,
                     max(128, cfg.max_total_reader_output_tokens - cell.get("output_used", 0)))
        return count + output + cfg.context_reserve_tokens + cfg.min_source_tokens < self.clients[backend].config.max_context_tokens

    def _research_budget(self, state: dict) -> dict:
        cfg = self.config.workflow
        return {"reopens": max(0, cfg.max_reopens - state["reopens"]),
                "searches": max(0, cfg.max_searches_per_query - sum(c["searches_used"] for c in state["cells"].values())),
                "document_fetches": max(0, cfg.max_document_fetches_per_query - sum(c["documents_used"] for c in state["cells"].values())),
                "reader_output_tokens": max(0, cfg.max_total_reader_output_tokens - sum(c["output_used"] for c in state["cells"].values())),
                "max_reader_turns_per_reopen": cfg.reopen_turns}

    def _research_options(self, state: dict) -> list[tuple[FinalDecision, dict]]:
        """Reuse actual pending leads first. No gold answers or extra LLM planner."""
        cfg = self.config.workflow
        if state["reopens"] >= cfg.max_reopens:
            return []
        cells = sorted(state["cells"].values(), key=lambda c: (-len(c.get("evidence", {})), c["cell_id"]))
        options = []
        for use_fallback in (False, True):
            for cell in cells:
                if cell.get("last_error"):
                    continue
                queries = (self._fallback_queries(cell) if use_fallback else cell.get("pending_queries", []))
                queries = unique_queries(queries, cell["seen_queries"], cfg.reopen_searches)
                reads = [] if use_fallback else cell.get("pending_read_more", [])
                if not queries and not reads:
                    continue
                decision = FinalDecision(action="research", reopen_cell_id=cell["cell_id"],
                    next_queries=queries, read_more=reads,
                    explanation="Resolve the remaining identifying or requested-value gap using the pending lead; "
                                "return the strongest grounded candidate, not a repeated checklist.")
                job = self._reopen_job(state, decision)
                if job:
                    options.append((decision, job))
            if options:
                break
        return options

    async def _recover_nonanswer(self, state: dict, updates: dict, selected: list[dict],
                                 terminal_retry: bool, reason: str, *, options: list,
                                 omitted: int) -> dict | None:
        if self.config.workflow.answer_policy != "best_effort" or terminal_retry:
            return None
        if options:
            decision, job = options[0]
            self.store.event("policy_reopen", scope=state["scope"], trigger=reason,
                             remaining_budget=self._research_budget(state), cell_id=job["id"],
                             next_queries=job["initial_queries"], read_more=job["read_more"])
            return {**updates, "decision": decision.model_dump(), "next_job": job,
                    "reopens": state["reopens"] + 1, "status": "researching"}
        # Review once only if an actual non-refuted supported candidate exists.
        # Do not select an answer in Python or modify the evidence validator.
        statuses = candidate_status(state["evidence"], state["constraints"])
        eligible = any(e["relation"] == "SUPPORTS" and
                       not statuses.get(e["candidate_key"], {}).get("has_required_contradiction", True)
                       for e in selected)
        if not eligible:
            return None
        self.store.event("terminal_choice_retry", scope=state["scope"], trigger=reason,
                         remaining_budget=self._research_budget(state))
        result = await self._terminal_remedy({**state, **updates}, selected, omitted,
                                             kind="terminal_choice", trigger=reason)
        updates.update(result)
        return updates if "decision" in result else None

    async def _terminal_remedy(self, state: dict, selected: list[dict], omitted: int, *,
                               kind: str, trigger: str) -> dict:
        """One durable allowance shared by format recovery and terminal review.

        The journal key lives outside model/tool operation scopes so it never
        counts as a model call. Committed HTTP calls still replay through ChatClient.
        """
        cfg = self.config.workflow
        key = "terminal-remedy:" + state["scope"]
        previous = self.store.get(key)
        record = {"used": False, "kind": kind, "trigger": trigger,
                  "source_round": state["decision_round"],
                  "evidence_hash": stable_hash(selected)}
        if state.get("terminal_remedy", {}).get("used") or (previous and previous["value"]["used"] and
                any(previous["value"].get(k) != record[k] for k in ("source_round", "kind", "evidence_hash"))):
            return self._skip_remedy(state, previous["value"] if previous else record, "already_used")
        charged = state["main_usage"][-1]["output_budget_charged"]
        # With the feature disabled, keep the original terminal review budget.
        limit = (min(cfg.final_recovery_tokens, cfg.final_output_tokens - charged)
                 if cfg.final_recovery_tokens else cfg.final_output_tokens)
        record["max_tokens"] = max(0, limit)
        if limit < 128:
            return self._skip_remedy(state, record, "insufficient_output_budget")
        try:
            result = await self.decide(state, _terminal_retry=True,
                                       _evidence_pack=(selected, omitted), _remedy=record)
        except BaseException:
            committed = self.store.get(key)
            if committed and committed["value"]["used"]:
                self.store.put(key, committed["fingerprint"], "success",
                               {**committed["value"], "outcome": "interrupted_or_unknown"})
            raise
        remedy = result.get("terminal_remedy", record)
        if remedy.get("used"):
            errors = result.get("errors", [])
            outcome = ("format_failed" if any(e.startswith("final_format_error:") for e in errors)
                       else "validation_failed" if any(e.startswith("final_validation_failed:") for e in errors)
                       else "service_error" if any(e.startswith("Main service error:") for e in errors)
                       else "research_rejected" if any(e.startswith("research_unavailable:") for e in errors)
                       else "answered" if result.get("status") == "completed"
                       else "abstained" if result.get("status") == "unresolved" else "unknown")
            remedy = {**remedy, "outcome": outcome,
                      "explanation": result.get("decision", {}).get("explanation", "")}
            committed = self.store.get(key)
            self.store.put(key, committed["fingerprint"], "success", remedy)
            result["terminal_remedy"] = remedy
            self.store.event("terminal_remedy_finished", scope=state["scope"], remedy=remedy)
        return result

    def _skip_remedy(self, state: dict, record: dict, reason: str) -> dict:
        record = {**record, "skip_reason": reason}
        self.store.event("terminal_remedy_skipped", scope=state["scope"], reason=reason, remedy=record)
        return {"terminal_remedy": record}

    async def decide(self, state: dict, *, _terminal_retry: bool = False,
                     _evidence_pack: tuple[list[dict], int] | None = None,
                     _remedy: dict | None = None) -> dict:
        cfg = self.config.workflow
        previous = self.store.get("terminal-remedy:" + state["scope"])
        spent = state.get("terminal_remedy", {})
        if not _terminal_retry and (spent.get("used") or (previous and previous["value"]["used"]
                and previous["value"]["source_round"] != state["decision_round"] + 1)):
            return {**self._unresolved(state, "terminal_remedy_already_used"),
                    "terminal_remedy": previous["value"] if previous else spent}
        rows = [e for e in state["evidence"].values() if e.get("active", True)]
        # Prioritize conflicts and target values. The coverage summary still shows
        # all candidates' known gaps/conflicts if the quotation pack is truncated.
        rows.sort(key=lambda e: (0 if e["relation"] == "CONTRADICTS" else 1 if e.get("answer_value") else 2,
                                 e["constraint_id"], e["evidence_id"]))
        options = [] if _terminal_retry else self._research_options(state)
        can_reopen = bool(options)
        budget = self._research_budget(state)
        prompt_policy = {"answer_policy": cfg.answer_policy,
                         "research_options": [{"cell_id": j["id"], "next_queries": j["initial_queries"],
                                               "read_more": j["read_more"]} for _, j in options],
                         "remaining_budget": budget,
                         "terminal_choice": bool(_remedy and _remedy["kind"] == "terminal_choice"),
                         "output_recovery": bool(_remedy and _remedy["kind"] == "output_recovery")}
        thinking = {}
        if cfg.decision_thinking == "phase":
            thinking["enable_thinking"] = can_reopen
        elif cfg.decision_thinking == "off":
            thinking["enable_thinking"] = False
        if _terminal_retry and cfg.final_recovery_tokens:
            thinking["enable_thinking"] = False
        max_tokens = (_remedy["max_tokens"] if _remedy else
                      cfg.final_output_tokens - cfg.final_recovery_tokens)
        counter = self.counters["main"]
        selected = []
        omitted = 0
        for row in (rows if _evidence_pack is None else []):
            trial = [*selected, row]
            if counter.text(js(trial)) > cfg.final_evidence_tokens:
                omitted += 1
            else:
                selected = trial
        if _evidence_pack is not None:
            selected, omitted = _evidence_pack
        message = final_message(state["question"], state["constraints"], selected,
                                state["candidates"], state["cells"], can_reopen, **prompt_policy)
        history = [*state["main_history"], message]
        cap = self.config.main.max_context_tokens - max_tokens - cfg.context_reserve_tokens
        while _evidence_pack is None and selected and counter.messages(history, **thinking) > cap:
            selected.pop()
            omitted += 1
            message = final_message(state["question"], state["constraints"], selected,
                                    state["candidates"], state["cells"], can_reopen, **prompt_policy)
            history = [*state["main_history"], message]
        count = counter.messages(history, **thinking)
        if count > cap:
            if _remedy:
                return self._skip_remedy(state, _remedy, "insufficient_context")
            return self._unresolved(state, "Main context exhausted; history was not silently rewritten")
        remaining = state["query_deadline"] - time.time() if state.get("query_deadline") is not None else None
        if remaining is not None and remaining <= 0:
            if _remedy:
                return self._skip_remedy(state, _remedy, "no_remaining_time")
            raise TimeoutError("Original query deadline expired before decision")
        if _remedy:
            _remedy = {**_remedy, "used": True, "remaining_seconds_at_start": remaining,
                       "query_deadline": state.get("query_deadline"), "outcome": "requested"}
            # Durable before dispatch: a failed node/checkpoint cannot mint a
            # second allowance, and ambiguous requests are never reissued.
            key = "terminal-remedy:" + state["scope"]
            fingerprint = previous["fingerprint"] if previous else stable_hash(_remedy)
            self.store.put(key, fingerprint, "success", _remedy)
            self.store.event("terminal_remedy_started", scope=state["scope"], remedy=_remedy)
        try:
            reply = await asyncio.wait_for(
                self.clients["main"].complete(
                    history, max_tokens,
                    operation_id=f'{state["scope"]}:main:decision:{state["decision_round"] + 1}',
                    writer_key=state["scope"] + ":main", local_prompt_tokens=count, **thinking),
                timeout=remaining)
        except ModelRequestError as exc:
            return {**self._unresolved(state, f"Main service error: {exc}"),
                    **({"terminal_remedy": _remedy} if _remedy else {})}
        actual_output = reply["usage"].get("completion_tokens")
        updates = {"main_history": [*history, reply["assistant"]],
                   "main_usage": [*state["main_usage"], {"stage": "decide", "usage": reply["usage"],
                       "kind": _remedy["kind"] if _remedy else "decision",
                       "requested_output_tokens": max_tokens,
                       "output_budget_charged": max_tokens if actual_output is None else actual_output,
                       "enable_thinking": reply.get("enable_thinking", thinking.get("enable_thinking", self.config.main.enable_thinking)),
                       "duration_ms": reply.get("duration_ms"), "finish_reason": reply.get("finish_reason")}],
                   "decision_round": state["decision_round"] + 1, "next_job": None}
        if _remedy:
            updates["terminal_remedy"] = _remedy
        failure_kind = "empty_content"
        try:
            body = message_text(reply["assistant"])
            if not body.strip():
                raise ValueError("Main response has no answer body")
            failure_kind = "schema_error"
            if reply["assistant"].get("tool_calls"):
                raise ValueError("Main returned unexpected tool calls")
            failure_kind = "json_error"
            raw_decision = parse_json_object(body)
            # E-numbers are scoped to this exact selected evidence pack. Do not
            # resolve an omitted item or guess an unknown hash from a prefix.
            alias_map = {v: k for k, v in evidence_aliases({e["evidence_id"]: e for e in selected}).items()}
            if isinstance(raw_decision.get("evidence_ids"), list):
                raw_decision["evidence_ids"] = [alias_map.get(i, i) if isinstance(i, str) else i
                                                for i in raw_decision["evidence_ids"]]
            failure_kind = "schema_error"
            decision = FinalDecision.model_validate(raw_decision)
        except ValueError as exc:
            updates["final_format_failures"] = state.get("final_format_failures", 0) + 1
            self.store.event("final_output_failure", scope=state["scope"], failure_kind=failure_kind,
                             decision_round=updates["decision_round"], terminal_remedy=bool(_remedy))
            if cfg.final_recovery_tokens and not _terminal_retry:
                recovered = await self._terminal_remedy({**state, **updates}, selected, omitted,
                                                        kind="output_recovery", trigger=failure_kind)
                updates.update(recovered)
                if "decision" in recovered:
                    return updates
            updates.update(self._unresolved(state, "final_format_error: " + str(exc)[:800]))
            return updates
        if decision.action == "research":
            job = (self._reopen_job(state, decision)
                   if decision.reopen_cell_id in {j["id"] for _, j in options} else None)
            if job:
                updates.update(next_job=job, reopens=state["reopens"] + 1,
                               decision=decision.model_dump(), status="researching")
                return updates
            recovered = await self._recover_nonanswer(state, updates, selected, _terminal_retry, "research_unavailable",
                                                      options=options, omitted=omitted)
            if recovered is not None:
                return recovered
            # No fabricated answer and no recursive retry loop. It is a budget/
            # lead-limited abstention, not an HTTP/JSON failure.
            updates.update(self._unresolved(state, "research_unavailable: no valid remaining budget or new lead"))
            self.store.event("research_rejected", scope=state["scope"], can_reopen=can_reopen,
                             reason="budget_or_no_new_lead")
            return updates
        if decision.action == "answer":
            context_ids: list[str] = []
            try:
                delivery, context_ids = partition_decision_citations(
                    decision.model_dump(), state["evidence"],
                    {r["evidence_id"] for r in selected})
                # Validate against the ENTIRE ledger. Context is never relabelled
                # or used to satisfy this candidate's coverage/target checks.
                valid, reason = validate_final(delivery, state["evidence"], state["constraints"],
                                               cfg.require_full_coverage_for_final)
            except ValueError as exc:
                valid, reason = False, str(exc)
            if valid:
                committed = delivery
                committed["proposed_evidence_ids"] = list(decision.evidence_ids)
                committed["context_evidence_ids"] = context_ids
                committed["context_citation_docids"] = list(dict.fromkeys(
                    state["evidence"][ident]["docid"] for ident in context_ids))
                cited = set(committed["evidence_ids"])
                committed["citation_docids"] = list(dict.fromkeys(
                    state["evidence"][ident]["docid"] for ident in committed["evidence_ids"]))
                committed["citation_aliases"] = {
                    alias: state["evidence"][ident]["docid"]
                    for alias, ident in alias_map.items() if ident in cited}
                committed["citation_aliases"].update({
                    ident: state["evidence"][ident]["docid"] for ident in committed["evidence_ids"]})
                updates.update(decision=committed, status="completed", answer_support=reason)
                if context_ids:
                    self.store.event("final_citations_partitioned", scope=state["scope"],
                                     proposed_evidence_ids=decision.evidence_ids,
                                     supporting_evidence_ids=committed["evidence_ids"],
                                     context_evidence_ids=context_ids)
            else:
                updates.update(self._unresolved(state, "final_validation_failed: " + reason))
        else:
            recovered = await self._recover_nonanswer(state, updates, selected, _terminal_retry, "model_unresolved",
                                                      options=options, omitted=omitted)
            if recovered is not None:
                return recovered
            unresolved = decision.model_dump()
            unresolved.update(exact_answer="", evidence_ids=[])
            updates.update(decision=unresolved, status="unresolved", answer_support="none")
        self.store.event("final_decision", scope=state["scope"], action=decision.action,
                         status=updates["status"], omitted_evidence_items=omitted,
                         answer_policy=cfg.answer_policy, terminal_choice=prompt_policy["terminal_choice"],
                         output_recovery=prompt_policy["output_recovery"])
        return updates

    def _unresolved(self, state: dict, reason: str) -> dict:
        return {"decision": {"action": "unresolved", "candidate": "", "exact_answer": "",
                             "explanation": reason, "evidence_ids": []},
                "status": "unresolved", "answer_support": "none", "next_job": None,
                "errors": [*state.get("errors", []), reason]}

    def render(self, state: dict) -> dict:
        return {"final_text": render_answer(state["decision"])}

    async def release_chains(self, state: dict) -> dict:
        """Release final Reader chains; reopened cells retain them until this node."""
        seen: set[tuple[str, str]] = set()
        released, errors = 0, []
        clients_by_endpoint = {client.url: client for client in self.clients.values()}
        for cell in state.get("cells", {}).values():
            handle = cell.get("handle") or {}
            key = (str(handle.get("endpoint") or ""), str(handle.get("chain_id") or ""))
            if not all(key) or key in seen:
                continue
            seen.add(key)
            client = clients_by_endpoint.get(key[0])
            if client is None:
                errors.append(f"No client owns Reader chain endpoint {key[0]}")
                continue
            try:
                await client.release_chain(handle)
                released += 1
                self.store.event("chain_released", scope=state["scope"], endpoint=key[0], chain_id=key[1])
            except Exception as exc:
                errors.append(f"{key[1]}: {type(exc).__name__}: {exc}")
                self.store.event("chain_release_error", scope=state["scope"], endpoint=key[0],
                                 chain_id=key[1], error=str(exc)[:800])
        updates = {"released_chains": released}
        if errors:
            updates["errors"] = [*state.get("errors", []), *("chain_release_error: " + e for e in errors)]
        return updates

    async def close(self):
        for client in self.clients.values():
            await client.close()
        await self.retrieval.close()
