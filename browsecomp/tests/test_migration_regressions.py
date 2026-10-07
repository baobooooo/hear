"""Migration-only regressions. Synthetic facts; no benchmark gold in runtime code."""
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from bcgraph.schemas import parse_plan_output, ReaderReply
from bcgraph.delivery import partition_decision_citations
from bcgraph.evidence import accept_reply, source_record, validate_final, candidate_key
from bcgraph.app import export_result
from bcgraph.demo import make_demo_runtime


def multicell_plan():
    return {"constraints": [
        {"id": "c1", "description": "Director in 2007"},
        {"id": "c2", "description": "Instrument designer"},
        {"id": "target", "description": "Requested name", "answer_target": True}],
        "cells": [
            {"id": "a", "focus": "Check directorship", "constraint_ids": ["c1"], "initial_queries": ["director 2007"]},
            {"id": "b", "focus": "Check designer", "constraint_ids": ["c2"], "initial_queries": ["instrument designer"]}]}


def test_single_deployed_cell_is_normalized_before_assignment_validation():
    raw = json.dumps(multicell_plan())
    plan, notes = parse_plan_output(raw, max_cells=1)
    assert len(plan.cells) == 1
    assert plan.cells[0].constraint_ids == ["c1", "c2", "target"]
    assert plan.cells[0].initial_queries == ["director 2007", "instrument designer"]
    assert len(plan.constraints) == 3
    assert notes == ["merged_planner_cells_before_assignment_validation",
                     "assigned_missing_constraints_to_only_cell:target"]
    assert json.loads(raw) == multicell_plan()


@pytest.mark.parametrize("max_cells", [None, 2, 3, 4])
def test_missing_multicell_assignment_remains_ambiguous_without_single_owner(max_cells):
    with pytest.raises(ValueError, match="Required constraints"):
        parse_plan_output(json.dumps(multicell_plan()), max_cells=max_cells)


@pytest.mark.parametrize("problem", ["unknown", "duplicate_cell", "duplicate_constraint", "no_target", "extra", "many"])
def test_normalization_does_not_hide_invalid_content(problem):
    obj = multicell_plan()
    if problem == "unknown": obj["cells"][1]["constraint_ids"].append("c999")
    elif problem == "duplicate_cell": obj["cells"][1]["id"] = "a"
    elif problem == "duplicate_constraint": obj["constraints"][1]["id"] = "c1"
    elif problem == "no_target": obj["constraints"][-1]["answer_target"] = False
    elif problem == "extra": obj["confidence"] = 99
    elif problem == "many": obj["cells"] *= 3
    with pytest.raises(ValueError): parse_plan_output(json.dumps(obj), max_cells=1)


def test_normalization_does_not_accept_multiple_outputs_or_truncated_json():
    raw = json.dumps(multicell_plan())
    for broken in (raw + raw, raw[:-10]):
        with pytest.raises(ValueError): parse_plan_output(broken, max_cells=1)


@pytest.mark.asyncio
async def test_runtime_uses_deployed_topology_and_preserves_raw_planner_response(store):
    runtime, _, http = make_demo_runtime(store)
    raw = json.dumps(multicell_plan())
    runtime.clients["main"].complete = AsyncMock(return_value={
        "assistant": {"role": "assistant", "content": raw}, "usage": {}})
    try:
        result = await runtime.plan({"question": "Synthetic question", "query_id": "test", "scope": "test"})
        assert set(result["constraints"]) == {"c1", "c2", "target"}
        assert len(result["jobs"]) == 1
        assert result["main_history"][-1]["content"] == raw
        assert not any(e.startswith("planner_format_fallback:") for e in result["errors"])
        assert runtime.clients["main"].complete.await_count == 1
    finally:
        await runtime.close(); await http.aclose()


def evidence_fixture():
    constraints = {"c1": {"id": "c1", "description": "Institution", "time_scope": "unspecified", "required": True, "answer_target": False},
                   "target": {"id": "target", "description": "Name", "time_scope": "unspecified", "required": True, "answer_target": True}}
    text = "Eira Example founded the Helix Institute. The Helix Institute opened in 1990."
    source = source_record("doc1", text, 0, len(text))
    raw = [{"candidate": "Eira Example", "constraint_id": "target", "source_id": "S1", "quote": "Eira Example founded the Helix Institute."},
           {"candidate": "Helix Institute", "constraint_id": "c1", "source_id": "S1", "quote": "The Helix Institute opened in 1990."}]
    ledger, errors, _ = accept_reply(ReaderReply(evidence=raw), {source["source_id"]: source}, constraints, {}, "cell1", 1)
    assert not errors
    decision = {"action": "answer", "candidate": "Eira Example", "exact_answer": "Eira Example", "evidence_ids": list(ledger)}
    return constraints, ledger, decision


def test_same_span_ancillary_entity_is_context_not_answer_support():
    constraints, ledger, decision = evidence_fixture()
    old_ledger, old_decision = deepcopy(ledger), deepcopy(decision)
    support_only, context = partition_decision_citations(decision, ledger, list(ledger))
    assert len(support_only["evidence_ids"]) == len(context) == 1
    assert validate_final(support_only, ledger, constraints)[0]
    assert not validate_final(support_only, ledger, constraints, True)[0]
    assert ledger == old_ledger and decision == old_decision
    assert next(e for e in ledger.values() if e["candidate"] == "Helix Institute")["candidate_key"] == "helix institute"


@pytest.mark.parametrize("problem", ["unknown", "omitted", "retracted", "different_span", "context_contradiction", "only_context"])
def test_context_partition_does_not_launder_unusable_citations(problem):
    _, ledger, decision = evidence_fixture()
    own, context = list(ledger)
    visible = list(ledger)
    if problem == "unknown": decision["evidence_ids"].append("missing")
    elif problem == "omitted": visible.remove(context)
    elif problem == "retracted": ledger[context]["active"] = False
    elif problem == "different_span": ledger[context]["source_id"] = "different-source-same-document"
    elif problem == "context_contradiction": ledger[context]["relation"] = "CONTRADICTS"
    elif problem == "only_context": decision["evidence_ids"] = [context]
    with pytest.raises(ValueError): partition_decision_citations(decision, ledger, visible)


def test_chosen_candidate_contradiction_outside_prompt_is_not_removed():
    constraints, ledger, decision = evidence_fixture()
    visible = list(ledger)
    conflict = deepcopy(ledger[visible[0]])
    conflict.update(evidence_id="conflict", relation="CONTRADICTS", constraint_id="target")
    ledger["conflict"] = conflict
    normalized, _ = partition_decision_citations(decision, ledger, visible)
    valid, reason = validate_final(normalized, ledger, constraints)
    assert not valid and "contradiction" in reason


@pytest.mark.parametrize("answer", ["Different Person", "Exampleton", "ira"])
def test_partition_does_not_supply_an_answer_or_allow_partial_name(answer):
    constraints, ledger, decision = evidence_fixture()
    decision["exact_answer"] = answer
    normalized, _ = partition_decision_citations(decision, ledger, list(ledger))
    assert normalized["exact_answer"] == answer
    assert not validate_final(normalized, ledger, constraints)[0]


@pytest.mark.asyncio
async def test_decide_keeps_original_response_and_separates_context_metadata(store):
    constraints, ledger, proposal = evidence_fixture()
    runtime, _, http = make_demo_runtime(store)
    runtime.config.workflow.require_full_coverage_for_final = False
    # Runtime sorts c1 then target. E1 is the institute, E2 the person.
    proposal["evidence_ids"] = ["E1", "E2"]
    raw = json.dumps(proposal)
    runtime.clients["main"].complete = AsyncMock(return_value={
        "assistant": {"role": "assistant", "content": raw}, "usage": {}})
    state = {"question": "Who founded Helix?", "scope": "delivery", "constraints": constraints,
             "evidence": ledger, "candidates": {}, "cells": {}, "reopens": 0, "decision_round": 0,
             "main_usage": [], "main_history": [], "errors": []}
    try:
        result = await runtime.decide(state)
        assert result["status"] == "completed"
        assert result["main_history"][-1]["content"] == raw
        assert result["decision"]["exact_answer"] == proposal["exact_answer"]
        assert len(result["decision"]["context_evidence_ids"]) == 1
        assert len(result["decision"]["evidence_ids"]) == 1
        assert len(result["decision"]["proposed_evidence_ids"]) == 2
        assert result["decision"]["citation_docids"] == ["doc1"]
    finally:
        await runtime.close(); await http.aclose()


def ready_cell(reply_queries=None, *, turn=1, searches_used=1, read_more=None):
    constraints, _, _ = evidence_fixture()
    text = "Eira Example directed the institute. Eira Example is the requested name."
    src = source_record("doc", text, 0, len(text))
    evidence = [{"candidate": "Eira Example", "constraint_id": "c1", "source_id": "S1", "quote": "Eira Example directed the institute."},
                {"candidate": "Eira Example", "constraint_id": "target", "source_id": "S1", "quote": "Eira Example is the requested name.", "answer_value": "Eira Example"}]
    return {"scope": "stop", "cell_id": "cell1", "turn": turn, "constraints": constraints,
            "constraint_ids": list(constraints), "sources": {src["source_id"]: src}, "evidence": {},
            "reply": {"assistant": {"content": json.dumps({"evidence": evidence, "next_queries": reply_queries or [], "read_more": read_more or []})}, "finish_reason": "stop"},
            "pending_queries": ["unexecuted planner query"], "seen_queries": ["already searched"], "summary": "",
            "no_progress_turns": 0, "turn_limit": 3, "output_limit": 5000, "output_used": 200,
            "documents_used": 1, "document_limit": 20, "searches_used": searches_used, "search_limit": 8,
            "doc_catalog": {"doc": {"docid": "doc"}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("queries,turn,searches,read_more,expected", [
    (["verify the institution"], 1, 1, [], ""),
    ([], 1, 1, [], "evidence_sufficient"),
    (["already searched"], 1, 1, [], "evidence_sufficient"),
    (["verify the institution"], 3, 1, [], "turn_budget_exhausted"),
    (["verify the institution"], 1, 8, [], "evidence_sufficient"),
    ([], 1, 1, [{"docid": "doc", "offset": 0}], ""),
    ([], 1, 1, [{"docid": "unknown", "offset": 0}], "evidence_sufficient"),
])
async def test_explicit_executable_reader_followup_is_not_overridden(store, queries, turn, searches, read_more, expected):
    runtime, _, http = make_demo_runtime(store)
    try:
        result = runtime.cell_validate(ready_cell(queries, turn=turn, searches_used=searches, read_more=read_more))
        assert result["stop_reason"] == expected
    finally:
        await runtime.close(); await http.aclose()


def test_planner_fallback_is_reported_without_changing_judge_denominator():
    state = {"query_id": "q", "status": "unresolved", "decision": {"action": "unresolved"},
             "errors": ["planner_format_fallback: missing assignment", "final_validation_failed: no citation"]}
    result = export_result(state, "test", 10)
    assert result["status"] == "unresolved"
    assert result["metadata"]["execution_status"] == "degraded"
    assert result["metadata"]["planner_fallback_count"] == 1
    assert result["metadata"]["final_validation_rejection_count"] == 1
    assert result["metadata"]["outcome"] == "abstained"

@pytest.mark.parametrize('bad', [None, 42, 'constraints'])
def test_malformed_constraint_collection_is_schema_error_not_type_crash(bad):
    plan = multicell_plan(); plan['constraints'] = bad
    with pytest.raises(ValueError): parse_plan_output(json.dumps(plan), max_cells=1)
