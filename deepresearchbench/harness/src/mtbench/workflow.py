from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .models import PlanTask, ResearchReport
from .services import WorkflowServices


class WorkflowState(TypedDict, total=False):
    instance_id: int
    query: str
    language: str
    plan: list[dict[str, Any]]
    reports: Annotated[list[dict[str, Any]], operator.add]
    current_round: int
    decision: str
    feedback: dict[str, dict[str, Any]]
    final_report: str


class ResearcherState(TypedDict):
    query: str
    task: dict[str, Any]
    round_no: int
    feedback: str | None
    selected_document_id: str | None
    search_query: str | None
    prior_report: dict[str, Any] | None
    report_history: list[dict[str, Any]]


def latest_reports(state: WorkflowState) -> dict[str, ResearchReport]:
    latest: dict[str, ResearchReport] = {}
    for payload in state.get("reports", []):
        report = ResearchReport.model_validate(payload)
        previous = latest.get(report.agent_id)
        if previous is None or report.round_no > previous.round_no:
            latest[report.agent_id] = report
    return latest


def build_workflow(services: WorkflowServices, *, report_rounds: int):
    if report_rounds < 1:
        raise ValueError("report_rounds must be at least 1")

    async def planner(state: WorkflowState) -> dict[str, Any]:
        tasks = await services.plan(state["query"], state["language"])
        return {
            "plan": [task.model_dump() for task in tasks],
            "current_round": 1,
            "decision": "research",
            "feedback": {},
        }

    def dispatch_plan(state: WorkflowState):
        return [
            Send(
                "researcher",
                {
                    "query": state["query"],
                    "task": task,
                    "round_no": 1,
                    "feedback": None,
                    "selected_document_id": None,
                    "search_query": None,
                    "prior_report": None,
                    "report_history": [],
                },
            )
            for task in state["plan"]
        ]

    async def researcher(state: ResearcherState) -> dict[str, Any]:
        report = await services.research(
            state["query"],
            PlanTask.model_validate(state["task"]),
            state["round_no"],
            state.get("feedback"),
            state.get("selected_document_id"),
            state.get("search_query"),
            ResearchReport.model_validate(state["prior_report"])
            if state.get("prior_report")
            else None,
            [ResearchReport.model_validate(item) for item in state.get("report_history", [])],
        )
        return {"reports": [report.model_dump()]}

    async def join_reports(state: WorkflowState) -> dict[str, Any]:
        current = state["current_round"]
        round_reports = [
            report
            for report in latest_reports(state).values()
            if report.round_no == current
        ]
        await services.trace.emit(
            "research_barrier_reached",
            round_no=current,
            report_count=len(round_reports),
            agent_ids=sorted(report.agent_id for report in round_reports),
        )
        return {}

    async def review(state: WorkflowState) -> dict[str, Any]:
        current = state["current_round"]
        reports = list(latest_reports(state).values())
        decision = await services.review(state["query"], reports, current)
        expected_ids = {task["agent_id"] for task in state["plan"]}
        feedback_items = {item.agent_id: item for item in decision.feedback}
        missing = expected_ids - feedback_items.keys()
        extra = feedback_items.keys() - expected_ids
        if missing or extra:
            raise RuntimeError(
                f"main feedback contract failed in round {current}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if current == 1:
            document_counts = {
                report.agent_id: len(report.documents[:5]) for report in reports
            }
            invalid = sorted(
                agent_id
                for agent_id, item in feedback_items.items()
                if not (
                    (item.document_id and item.document_id.startswith("D"))
                    or (
                        document_counts[agent_id] == 0
                        and item.search_query
                        and item.search_query.strip()
                    )
                )
            )
            if invalid:
                raise RuntimeError(
                    f"round-one feedback omitted a valid document_id for {invalid}"
                )
        if current >= 2 and current < report_rounds:
            invalid = sorted(
                agent_id
                for agent_id, item in feedback_items.items()
                if not item.search_query or not item.search_query.strip()
            )
            if invalid:
                raise RuntimeError(
                    f"round-two feedback omitted a counterevidence search_query for {invalid}"
                )
        feedback = {
            agent_id: item.model_dump() for agent_id, item in feedback_items.items()
        }
        must_continue = current < report_rounds
        enforced_decision = "revise" if must_continue else "finalize"
        await services.trace.emit(
            "fixed_round_route_enforced",
            round_no=current,
            configured_rounds=report_rounds,
            reviewer_decision=decision.decision,
            enforced_decision=enforced_decision,
        )
        return {
            "decision": enforced_decision,
            "feedback": feedback,
            "current_round": current + 1 if must_continue else current,
        }

    def route_after_review(state: WorkflowState):
        if state["decision"] != "revise":
            return "writer"
        prior = latest_reports(state)
        histories: dict[str, list[ResearchReport]] = {}
        for payload in state.get("reports", []):
            report = ResearchReport.model_validate(payload)
            histories.setdefault(report.agent_id, []).append(report)
        for reports in histories.values():
            reports.sort(key=lambda report: report.round_no)
        tasks = {task["agent_id"]: task for task in state["plan"]}
        return [
            Send(
                "researcher",
                {
                    "query": state["query"],
                    "task": tasks[agent_id],
                    "round_no": state["current_round"],
                    "feedback": state["feedback"][agent_id]["instruction"],
                    "selected_document_id": state["feedback"][agent_id].get(
                        "document_id"
                    ),
                    "search_query": state["feedback"][agent_id].get("search_query"),
                    "prior_report": prior[agent_id].model_dump(),
                    "report_history": [report.model_dump() for report in histories[agent_id]],
                },
            )
            for agent_id in tasks
        ]

    async def writer(state: WorkflowState) -> dict[str, Any]:
        reports = list(latest_reports(state).values())
        final = await services.write(
            state["query"], reports, state["language"], state.get("feedback", {}),
            history_reports=[
                ResearchReport.model_validate(payload)
                for payload in state.get("reports", [])
            ],
        )
        return {"final_report": final}

    graph = StateGraph(WorkflowState)
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("join_reports", join_reports)
    graph.add_node("review", review)
    graph.add_node("writer", writer)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch_plan, ["researcher"])
    graph.add_edge("researcher", "join_reports")
    graph.add_edge("join_reports", "review")
    graph.add_conditional_edges("review", route_after_review, ["researcher", "writer"])
    graph.add_edge("writer", END)
    return graph.compile()
