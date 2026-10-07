"""Bounded policy regressions; synthetic facts, no benchmark gold."""
from copy import deepcopy
import json
from unittest.mock import AsyncMock
import pytest
from bcgraph.config import WorkflowConfig
from bcgraph.demo import make_demo_runtime, QUESTION
from bcgraph.evidence import source_record, candidate_status, normalize_query
from bcgraph.prompts import final_message
from bcgraph.schemas import FinalDecision
from test_migration_regressions import evidence_fixture, ready_cell


def parent(runtime, *, evidence=False):
    constraints, ledger, _ = evidence_fixture()
    state = {"question": "Who founded Helix?", "query_id": "test", "scope": "policy",
             "constraints": constraints, "evidence": ledger if evidence else {},
             "candidates": candidate_status(ledger, constraints) if evidence else {},
             "cells": {}, "reopens": 0, "decision_round": 0,
             "main_usage": [], "main_history": [], "errors": []}
    job = {"id": "cell1", "focus": "Identify the founder", "constraint_ids": list(constraints),
           "initial_queries": ["Helix founder"], "search_limit": 8, "document_limit": 26,
           "output_limit": 5976, "turn_limit": 3}
    cell = runtime.init_cell({**state, "job": job, "previous": None})
    cell.update(turn=3, searches_used=8, documents_used=17, output_used=1200,
                stop_reason="turn_budget_exhausted", seen_queries=[normalize_query("Helix founder")],
                pending_queries=["Helix founder biography"], backend="reader")
    if evidence:
        cell["evidence"] = deepcopy(ledger)
    state["cells"] = {"cell1": cell}
    return state


def reply(data):
    return {"assistant": {"role": "assistant", "content": json.dumps(data)},
            "usage": {"completion_tokens": 100}}


def test_defaults_preserve_conservative_baseline():
    assert WorkflowConfig().answer_policy == "conservative"
    with pytest.raises(ValueError): WorkflowConfig(answer_policy="best_effort", require_full_coverage_for_final=True)


def test_best_effort_prompt_separates_uncertainty_from_refutation():
    p = final_message("Question", {}, [], {}, {}, True, answer_policy="best_effort",
                      remaining_budget={"searches": 2})["content"]
    assert "secondary clue" in p and "BEFORE unresolved" in p
    assert '"explanation":"The remaining factual gap."' not in p
    assert json.loads(p.rsplit("\n", 1)[1])["remaining_budget"]["searches"] == 2
    p = final_message("Question", {}, [], {}, {}, False, answer_policy="best_effort",
                      terminal_choice=True)["content"]
    assert "single terminal-choice review" in p and "Do not output action=research" in p


@pytest.mark.asyncio
async def test_unresolved_reopens_pending_lead_without_new_planner_or_answer(store):
    r, _, http = make_demo_runtime(store); r.config.workflow.answer_policy="best_effort"
    state = parent(r); original=deepcopy(state)
    response=reply({"action":"unresolved", "explanation":"The founder biography is not yet verified."})
    r.clients["main"].complete = AsyncMock(return_value=response)
    try:
        out = await r.decide(state)
        assert out["status"] == "researching" and out["reopens"] == 1
        assert out["next_job"]["initial_queries"] == ["Helix founder biography"]
        assert out["next_job"]["search_limit"] == 10
        assert out["main_history"][-1] == response["assistant"]
        assert out["decision"]["exact_answer"] == "" and state==original
        assert r.clients["main"].complete.await_count == 1
    finally: await r.close(); await http.aclose()


@pytest.mark.asyncio
async def test_conservative_keeps_model_abstention(store):
    r, _, http = make_demo_runtime(store)
    r.clients["main"].complete = AsyncMock(return_value=reply({"action":"unresolved","explanation":"gap"}))
    try:
        out=await r.decide(parent(r))
        assert out["status"]=="unresolved" and out["next_job"] is None
        assert r.clients["main"].complete.await_count==1
    finally: await r.close(); await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("exhaustion",["reopens","docs","output","reader_error","context"])
async def test_auto_reopen_respects_resource_limits(store,exhaustion):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r)
    c=state["cells"]["cell1"]
    if exhaustion=="reopens":state["reopens"]=1
    elif exhaustion=="docs":c["documents_used"]=r.config.workflow.max_document_fetches_per_query
    elif exhaustion=="output":c["output_used"]=r.config.workflow.max_total_reader_output_tokens
    elif exhaustion=="reader_error":c["last_error"]="request timed out"
    else:c["raw_history"].append({"role":"user","content":"z"*100000})
    try:assert r._research_options(state)==[]
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_known_read_more_survives_reopen_when_search_budget_is_zero(store):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r)
    c=state["cells"]["cell1"];c["searches_used"]=r.config.workflow.max_searches_per_query
    src=source_record("doc","First passage. Second passage.",0,14)
    c["sources"]={src["source_id"]:src};c["pending_queries"]=[]
    c["pending_read_more"]=[{"docid":"doc","offset":15}]
    try:
        _,job=r._research_options(state)[0]
        assert job["initial_queries"]==[] and job["read_more"]==[{"docid":"doc","offset":15}]
        continued=r.init_cell({**state,"job":job,"previous":c})
        assert continued["pending_read_more"]==job["read_more"]
        assert continued["sources"]==c["sources"] and continued["raw_history"]==c["raw_history"]
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("doc,offset",[("unknown",0),("doc",999)])
async def test_unknown_or_out_of_range_read_does_not_consume_reopen(store,doc,offset):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r)
    c=state["cells"]["cell1"];src=source_record("doc","short text",0,10)
    c["sources"]={src["source_id"]:src}
    d=FinalDecision(action="research",reopen_cell_id="cell1",read_more=[{"docid":doc,"offset":offset}])
    try:assert r._reopen_job(state,d) is None
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_duplicate_queries_are_not_reissued(store):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r)
    c=state["cells"]["cell1"];c["pending_queries"]=["  Helix founder  ","Helix founder biography"]
    try:assert r._research_options(state)[0][1]["initial_queries"]==["Helix founder biography"]
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("last_action",["unresolved","answer","research"])
async def test_terminal_review_is_bounded_and_preserves_both_raw_replies(store,last_action):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r,evidence=True)
    state["reopens"]=1
    final={"action":last_action,"explanation":"No new source is available."}
    if last_action=="answer":
        final.update(candidate="Eira Example",exact_answer="Eira Example",evidence_ids=[next(iter(state["evidence"]))])
    if last_action=="research":final.update(reopen_cell_id="cell1",next_queries=["new query"])
    responses=[reply({"action":"unresolved","explanation":"Some clues remain unchecked."}),reply(final)]
    r.clients["main"].complete=AsyncMock(side_effect=responses)
    try:
        out=await r.decide(state)
        assert r.clients["main"].complete.await_count==2 and out["decision_round"]==2
        assert len(out["main_usage"])==2
        assert [m for m in out["main_history"] if m["role"]=="assistant"]==[x["assistant"] for x in responses]
        assert out["status"]==("completed" if last_action=="answer" else "unresolved")
        if last_action=="answer":assert out["answer_support"]=="answer_with_gaps"
        assert r.clients["main"].complete.await_args_list[0].kwargs["operation_id"] != r.clients["main"].complete.await_args_list[1].kwargs["operation_id"]
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_empty_evidence_never_becomes_a_forced_answer_or_endless_retry(store):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r)
    state["reopens"]=1
    r.clients["main"].complete=AsyncMock(return_value=reply({"action":"unresolved","explanation":"No date in sources."}))
    try:
        out=await r.decide(state)
        assert out["status"]=="unresolved" and out["decision"]["exact_answer"]==""
        assert r.clients["main"].complete.await_count==1
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_required_contradiction_is_not_relaxed_by_best_effort(store):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort";state=parent(r,evidence=True)
    first=next(iter(state["evidence"].values()))
    state["evidence"]["conflict"]={**first,"evidence_id":"conflict","relation":"CONTRADICTS"}
    r.clients["main"].complete=AsyncMock(return_value=reply({"action":"answer","candidate":"Eira Example",
        "exact_answer":"Eira Example","evidence_ids":[first["evidence_id"]]}))
    try:
        out=await r.decide(state)
        assert out["status"]=="unresolved" and "contradiction" in out["decision"]["explanation"]
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior,queries,turn,docs,expected",[
    (1,["new fact query"],2,10,""),
    (2,["another query"],3,10,"no_evidence_progress"),
    (1,[],2,10,"no_evidence_progress"),
    (1,["already searched"],2,10,"no_evidence_progress"),
    (1,["new fact query"],5,10,"turn_budget_exhausted"),
    (1,["new fact query"],2,20,"document_budget_exhausted"),
])
async def test_no_progress_has_at_most_one_fresh_lead_grace_round(store,prior,queries,turn,docs,expected):
    r,_,http=make_demo_runtime(store);r.config.workflow.answer_policy="best_effort"
    c=ready_cell();c.update(turn_limit=5,turn=turn,documents_used=docs,no_progress_turns=prior,pending_queries=[])
    c["reply"]["assistant"]["content"]=json.dumps({"evidence":[],"next_queries":queries})
    try:assert r.cell_validate(c)["stop_reason"]==expected
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_reopen_reserves_two_rounds_of_documents_and_preserves_global_caps(store):
    r,_,http=make_demo_runtime(store);cfg=r.config.workflow
    cfg.answer_policy="best_effort";cfg.reopen_turns=2;cfg.reopen_searches=4
    cfg.max_reader_turns=5;cfg.max_searches_per_query=16;cfg.max_document_fetches_per_query=42
    cfg.max_total_reader_output_tokens=10000
    try:
        planned=await r.plan({"question":QUESTION,"query_id":"p","scope":"planbudgets"})
        j=planned["jobs"][0]
        assert j["search_limit"]==12 and j["document_limit"]==34 and j["output_limit"]==7952
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_no_document_budget_does_not_make_unusable_search_calls(store):
    r,_,http=make_demo_runtime(store);state=parent(r);c=state["cells"]["cell1"]
    c.update(turn=1,documents_used=c["document_limit"])
    r.retrieval.search=AsyncMock()
    try:
        result=await r.cell_fetch(c)
        assert result["stop_reason"]=="document_budget_exhausted"
        assert r.retrieval.search.await_count==0
    finally:await r.close();await http.aclose()


@pytest.mark.asyncio
async def test_real_node_cycle_reopens_same_chain_and_answers_without_filling_all_rounds(store):
    """Actual node + HTTP mock + chain protocol, NOT a LangGraph engine test."""
    r,backend,http=make_demo_runtime(store,chain=True)
    r.config.workflow.answer_policy="best_effort";r.config.workflow.max_reader_turns=1
    r.config.workflow.reopen_turns=2
    state={"question":QUESTION,"query_id":"cycle","scope":"cycle","attempt":1}
    try:
        state.update(await r.plan(state));c=r.init_cell({**state,"job":state["jobs"][0],"previous":None})
        c.update(await r.cell_fetch(c));c.update(await r.cell_read(c));c.update(r.cell_validate(c))
        assert c["stop_reason"]=="turn_budget_exhausted"
        state["cells"]={"cell1":c};state.update(r.collect(state));state.update(await r.decide(state))
        assert state["status"]=="researching" and state["reopens"]==1
        previous=deepcopy(c);c=r.init_cell({**state,"job":state["next_job"],"previous":c})
        assert c["handle"]==previous["handle"] and c["raw_history"]==previous["raw_history"]
        c.update(await r.cell_fetch(c));c.update(await r.cell_read(c));c.update(r.cell_validate(c))
        assert c["turn"]==2 and c["stop_reason"]=="evidence_sufficient"
        assert c["metrics"][-1]["request_mode"]=="chain_delta"
        state["cells"]={"cell1":c};state.update(r.collect(state));state.update(await r.decide(state))
        assert state["status"]=="completed" and state["decision"]["exact_answer"]=="Aurora spectrograph"
        assert len(state["main_usage"])==3  # plan, initial non-answer, post-research answer
    finally:await r.close();await http.aclose()
