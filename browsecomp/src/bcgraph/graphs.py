"""Actual LangGraph topology; no handwritten replacement for the graph engine.

A child invocation owns an entire research cell lifecycle, including its local
search/read loop. Parallel cells synchronize only when they return to collect.
"""
from __future__ import annotations
from typing import Annotated, Any, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from .evidence import merge_cells
from .runtime import Runtime


class CellState(TypedDict, total=False):
    delivery_original_reply: dict
    fetch_route: str
    last_consumed_feedback_revision: int
    selection_operation_id: str
    request_seq: int
    expanded_research_reports: bool
    stage_until: int
    selection_next_queries: list[str]
    research_directions: bool
    research_direction_guards: bool
    reader_mode: str
    latest_research_report: str
    report_status: str
    report_turn: int
    report_passage_refs: dict
    protocol: str
    passage_chars: int
    selected_passages: dict
    candidate_answer: str
    missing_fact: str
    selection_turns: list[dict]
    query_id: str
    question: str
    scope: str
    cell_id: str
    revision: int
    focus: str
    constraint_ids: list[str]
    constraints: dict
    search_limit: int
    document_limit: int
    output_limit: int
    turn_limit: int
    pending_queries: list[str]
    pending_read_more: list[dict]
    raw_history: list[dict]
    handle: dict | None
    backend: str
    sources: dict
    evidence: dict
    doc_catalog: dict
    seen_queries: list[str]
    searches_used: int
    documents_used: int
    output_used: int
    requested_output_tokens: int
    turn: int
    no_progress_turns: int
    stop_reason: str
    summary: str
    validation_errors: list[str]
    retrieval_notes: list[str]
    metrics: list[dict]
    last_error: str
    packed: dict
    reply: dict
    reopen_instruction: str
    reader_format_failures: int
    reader_partial_recoveries: int
    reader_delivery_repairs: int
    protocol_issues: list[str]
    retrieval_context: str


class ParentState(TypedDict, total=False):
    next_jobs: list[dict]
    sync_round: int
    protocol: str
    final_repair_count: int
    final_review_count: int
    delivery_issues: list[str]
    decision_exit_cause: str
    query_id: str
    question: str
    scope: str
    attempt: int
    query_deadline: float
    terminal_remedy: dict
    final_format_failures: int
    constraints: dict
    jobs: list[dict]
    main_history: list[dict]
    main_usage: list[dict]
    cells: Annotated[dict, merge_cells]
    evidence: dict
    candidates: dict
    decision_round: int
    reopens: int
    decision: dict
    next_job: dict | None
    final_text: str
    status: str
    answer_support: str
    errors: list[str]


class WorkerInput(TypedDict):
    query_id: str
    question: str
    scope: str
    constraints: dict
    job: dict
    previous: dict | None


def build_graph(runtime: Runtime, checkpointer: Any = None):
    cell = StateGraph(CellState)
    cell.add_node("retrieve_and_pack", runtime.cell_fetch)
    cell.add_node("read", runtime.cell_read)
    cell.add_node("validate", runtime.cell_validate)
    cell.add_edge(START, "retrieve_and_pack")
    cell.add_conditional_edges("retrieve_and_pack", lambda s: END if s.get("stop_reason") or s.get("fetch_route") == "yield"
                               else "retrieve_and_pack" if s.get("fetch_route") == "acquire" else "read")
    cell.add_conditional_edges("read", lambda s: END if s.get("stop_reason") else "validate")
    cell.add_conditional_edges("validate", lambda s: END if s.get("stop_reason") else "retrieve_and_pack")
    # Parent checkpoints are durable. Children are invocation-isolated and replay
    # committed HTTP/tool operations from the journal if the parent node restarts.
    # This avoids sharing one persistent child namespace across dynamic Send tasks.
    child = cell.compile(checkpointer=False)

    async def research_cell(state: WorkerInput, config: RunnableConfig) -> dict:
        initial = runtime.init_cell(state)
        async with runtime.live_cells.slot(continuation=bool(state.get("previous"))):
            result = await child.ainvoke(initial, config=config)
        # Return only one snapshot, never a cumulative list delta every child turn.
        result.pop("packed", None)
        result.pop("reply", None)
        return {"cells": {result["cell_id"]: result}}

    def work_input(state: ParentState, job: dict, previous: dict | None = None) -> WorkerInput:
        return {"query_id": state["query_id"], "question": state["question"], "scope": state["scope"],
                "constraints": state["constraints"], "job": job, "previous": previous}

    def dispatch(state: ParentState):
        return [Send("research_cell", work_input(state, job)) for job in state["jobs"]]

    def after_decide(state: ParentState):
        if state.get('next_jobs'):
            return [Send('research_cell', work_input(state, job, state['cells'][job['id']]))
                    for job in state['next_jobs']]
        if state.get("next_job"):
            job = state["next_job"]
            return Send("research_cell", work_input(state, job, state["cells"][job["id"]]))
        return "release_chains"

    graph = StateGraph(ParentState)
    graph.add_node("plan", runtime.plan)
    graph.add_node("research_cell", research_cell)
    graph.add_node("collect", runtime.collect)
    graph.add_node("decide", runtime.decide)
    graph.add_node("release_chains", runtime.release_chains)
    graph.add_node("render", runtime.render)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch, ["research_cell"])
    graph.add_edge("research_cell", "collect")
    graph.add_edge("collect", "decide")
    graph.add_conditional_edges("decide", after_decide, ["research_cell", "release_chains"])
    graph.add_edge("release_chains", "render")
    graph.add_edge("render", END)
    return graph.compile(checkpointer=checkpointer)
