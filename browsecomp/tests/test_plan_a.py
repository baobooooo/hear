"""Plan A uses synthetic evidence and the existing HTTP/graph test fixtures."""
import asyncio
from copy import deepcopy
import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from bcgraph.app import export_result
from bcgraph.config import EndpointConfig, WorkflowConfig
from bcgraph.demo import QUESTION, make_demo_runtime
from bcgraph.graphs import build_graph
from bcgraph.metrics import summarize
from bcgraph.storage import JournalConflict
from bcgraph.transport import ChatClient
from test_answer_policy import parent, reply


def enable_plan_a(runtime):
    runtime.config.main.enable_thinking = True
    runtime.config.main.max_context_tokens = 131072
    runtime.config.workflow.decision_thinking = "phase"
    runtime.config.workflow.final_output_tokens = 32768
    runtime.config.workflow.final_recovery_tokens = 1024
    runtime.config.workflow.answer_policy = "best_effort"


@pytest.fixture
async def plan_a(store):
    runtime, backend, http = make_demo_runtime(store)
    enable_plan_a(runtime)
    try:
        yield runtime
    finally:
        await runtime.close()
        await http.aclose()


def answer(state):
    evidence = next(e for e in state["evidence"].values() if e["candidate"] == "Eira Example")
    return {"action": "answer", "candidate": "Eira Example", "exact_answer": "Eira Example",
            "evidence_ids": [evidence["evidence_id"]], "explanation": "The quoted source names the founder."}


def broken(content=None, tokens=31744):
    return {"assistant": {"role": "assistant", "content": content,
                          "reasoning_content": "An unsupported guess is not evidence."},
            "usage": {"completion_tokens": tokens}, "finish_reason": "length"}


def prompt_data(call):
    return json.loads(call.args[0][-1]["content"].rsplit("\n", 1)[1])


@pytest.mark.parametrize("reserve", [1, 127, 32768, 32769])
def test_recovery_budget_validation(reserve):
    with pytest.raises(ValueError, match="final_recovery_tokens"):
        WorkflowConfig(final_output_tokens=32768, final_recovery_tokens=reserve)
    assert WorkflowConfig().final_recovery_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sparse-vllm", "deepseek"])
async def test_concurrent_thinking_overrides_are_request_local_and_journaled(store, engine):
    cfg = EndpointConfig(engine=engine, enable_thinking=True)
    original = cfg.model_dump()
    seen = []

    async def backend(request):
        seen.append(json.loads(request.content))
        await asyncio.sleep(0)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "{}"}}],
                                         "usage": {"completion_tokens": 1}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client = ChatClient(cfg, store, http=http)
        messages = [{"role": "user", "content": "test"}]
        results = await asyncio.gather(*(client.complete(messages, 128, operation_id=f"request-{flag}",
                        writer_key=str(flag), enable_thinking=flag) for flag in (True, False)))
        assert [r["enable_thinking"] for r in results] == [True, False]
        for body, flag in zip(seen, (True, False)):
            if engine == "deepseek":
                assert body["thinking"]["type"] == ("enabled" if flag else "disabled")
                assert "chat_template_kwargs" not in body
            else:
                assert body["chat_template_kwargs"]["enable_thinking"] is flag
                if engine == "sparse-vllm":
                    assert body["enable_thinking"] is flag
        replay = await client.complete(messages, 128, operation_id="request-False", writer_key="False", enable_thinking=False)
        assert replay["journal_replay"] and len(seen) == 2
        with pytest.raises(JournalConflict):
            await client.complete(messages, 128, operation_id="request-False", writer_key="False", enable_thinking=True)
    assert cfg.model_dump() == original


@pytest.mark.asyncio
async def test_final_chain_release_deduplicates_handles_and_uses_owner(plan_a):
    reader = plan_a.clients['reader']
    reader.release_chain = AsyncMock(return_value={'released': True})
    handle = {'chain_id': 'shared', 'endpoint': reader.url}
    state = {'scope': 'q1', 'cells': {'a': {'handle': handle}, 'b': {'handle': deepcopy(handle)}},
             'errors': []}
    out = await plan_a.release_chains(state)
    assert out == {'released_chains': 1}
    reader.release_chain.assert_awaited_once_with(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["phase", "off"])
@pytest.mark.parametrize("constraint", ["none", "reopens", "context", "no_leads", "documents", "output"])
async def test_decision_phase_uses_executable_options(plan_a, constraint, mode):
    plan_a.config.workflow.decision_thinking = mode
    state = parent(plan_a, evidence=True)
    cell = state["cells"]["cell1"]
    if constraint == "reopens":
        state["reopens"] = 1
    elif constraint == "context":
        cell["raw_history"].append({"role": "user", "content": "x" * 100000})
    elif constraint == "no_leads":
        cell["pending_queries"] = []
        plan_a._fallback_queries = lambda _: []
    elif constraint == "documents":
        cell["documents_used"] = plan_a.config.workflow.max_document_fetches_per_query
    elif constraint == "output":
        cell["output_used"] = plan_a.config.workflow.max_total_reader_output_tokens
    plan_a.clients["main"].complete = AsyncMock(return_value=reply(answer(state)))
    out = await plan_a.decide(state)
    call = plan_a.clients["main"].complete.await_args
    assert out["status"] == "completed"
    assert call.kwargs["enable_thinking"] is (mode == "phase" and constraint == "none")
    assert prompt_data(call)["can_reopen"] is (constraint == "none")
    assert call.args[1] == 31744


@pytest.mark.asyncio
@pytest.mark.parametrize("content,trigger", [(None, "empty_content"), ("{broken", "json_error"),
    ('{"action":"unresolved","cell_id":"cell1"}', "schema_error")])
async def test_format_recovery_keeps_evidence_mapping_history_budget_and_counters(plan_a, store, content, trigger):
    state = parent(plan_a, evidence=True)
    state["query_deadline"] = time.time() + 30
    original = deepcopy(state)
    first = broken(content)

    async def respond(messages, max_tokens, **kwargs):
        if plan_a.clients["main"].complete.await_count == 1:
            return first
        data = json.loads(messages[-1]["content"].rsplit("\n", 1)[1])
        target = next(e for e in data["validated_evidence"] if e["candidate"] == "Eira Example")
        return reply({**answer(state), "evidence_ids": [target["evidence_id"]]})

    plan_a.clients["main"].complete = AsyncMock(side_effect=respond)
    out = await plan_a.decide(state)
    calls = plan_a.clients["main"].complete.await_args_list
    assert state == original
    assert out["status"] == "completed" and out["decision_round"] == 2
    assert [c.args[1] for c in calls] == [31744, 1024]
    assert [c.kwargs["enable_thinking"] for c in calls] == [True, False]
    assert prompt_data(calls[0])["validated_evidence"] == prompt_data(calls[1])["validated_evidence"]
    assert prompt_data(calls[1])["output_recovery"] and not prompt_data(calls[1])["can_reopen"]
    assert calls[1].args[0][-2] == first["assistant"]
    assert out["terminal_remedy"]["trigger"] == trigger and out["terminal_remedy"]["used"]
    assert out["terminal_remedy"]["query_deadline"] == state["query_deadline"]
    assert sum(m["output_budget_charged"] for m in out["main_usage"]) <= 32768
    result = export_result({**state, **out}, "test", 1, store=store)
    assert result["metadata"]["execution_status"] == "degraded"
    assert result["metadata"]["outcome"] == "answered"
    assert result["metadata"]["final_format_failure_count"] == 1
    assert result["tool_call_counts"]["main"] == 2
    store.write_result(result)
    summary = summarize(store.path)
    assert summary["final_format_errors"] == 0
    assert summary["output_recovery_outcomes"] == {"answered": 1}
    assert summary["final_output_failure_counts"] == {trigger: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["abstain", "research", "bad_reference", "empty", "bad_schema"])
async def test_recovery_cannot_research_bypass_validation_or_retry_again(plan_a, second):
    state = parent(plan_a, evidence=True)
    value = {"action": "unresolved", "explanation": "No sufficiently supported answer."}
    if second == "research":
        value.update(action="research", reopen_cell_id="cell1", next_queries=["new lead"])
    elif second == "bad_reference":
        value = {**answer(state), "evidence_ids": ["E999"]}
    elif second == "bad_schema":
        value["cell_id"] = "cell1"
    plan_a.clients["main"].complete = AsyncMock(side_effect=[broken(), broken(tokens=1024) if second == "empty" else reply(value)])
    out = await plan_a.decide(state)
    assert out["status"] == "unresolved" and out["next_job"] is None
    assert plan_a.clients["main"].complete.await_count == 2
    assert out["terminal_remedy"]["used"] and out["terminal_remedy"]["kind"] == "output_recovery"
    expected = {"abstain": "abstained", "research": "research_rejected", "bad_reference": "validation_failed",
                "empty": "format_failed", "bad_schema": "format_failed"}
    assert out["terminal_remedy"]["outcome"] == expected[second]
    if second == "bad_reference":
        assert out["errors"][-1].startswith("final_validation_failed:")
    # The recorded allowance also protects a later branch with no flag in its state.
    await plan_a.decide({**state, "decision_round": 3})
    assert plan_a.clients["main"].complete.await_count == 2


@pytest.mark.asyncio
async def test_initial_bad_reference_does_not_trigger_recovery(plan_a):
    state = parent(plan_a, evidence=True)
    plan_a.clients["main"].complete = AsyncMock(return_value=reply({**answer(state), "evidence_ids": ["E999"]}))
    out = await plan_a.decide(state)
    assert out["status"] == "unresolved" and "terminal_remedy" not in out
    assert plan_a.clients["main"].complete.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["phase", "off"])
async def test_legal_abstention_uses_existing_research_policy(plan_a, mode):
    plan_a.config.workflow.decision_thinking = mode
    state = parent(plan_a)
    plan_a.clients["main"].complete = AsyncMock(return_value=reply({"action": "unresolved", "explanation": "No founder found."}))
    out = await plan_a.decide(state)
    assert out["status"] == "researching" and out["reopens"] == 1
    assert "terminal_remedy" not in out and plan_a.clients["main"].complete.await_count == 1
    assert plan_a.clients["main"].complete.await_args.kwargs["enable_thinking"] is (mode == "phase")
    state["reopens"] = 1
    out = await plan_a.decide(state)
    assert out["status"] == "unresolved" and "terminal_remedy" not in out


@pytest.mark.asyncio
async def test_existing_terminal_review_consumes_same_reserve(plan_a):
    state = parent(plan_a, evidence=True)
    state["reopens"] = 1
    plan_a.clients["main"].complete = AsyncMock(side_effect=[reply({"action": "unresolved", "explanation": "Some uncertainty remains."}), broken(tokens=1024)])
    out = await plan_a.decide(state)
    assert out["status"] == "unresolved" and out["terminal_remedy"]["kind"] == "terminal_choice"
    assert [c.args[1] for c in plan_a.clients["main"].complete.await_args_list] == [31744, 1024]
    assert all(c.kwargs["enable_thinking"] is False for c in plan_a.clients["main"].complete.await_args_list)
    assert out["final_format_failures"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["time", "context", "output"])
async def test_recovery_preflight_skips_without_an_extra_call(plan_a, monkeypatch, limit):
    state = parent(plan_a, evidence=True)
    now = [100.0]
    monkeypatch.setattr("bcgraph.runtime.time.time", lambda: now[0])
    state["query_deadline"] = 101.0

    async def respond(*args, **kwargs):
        value = broken(tokens=32768 if limit == "output" else 31744)
        if limit == "time":
            now[0] = 102.0
        if limit == "context":
            value["assistant"]["reasoning_content"] = "x" * 200000
        return value

    plan_a.clients["main"].complete = AsyncMock(side_effect=respond)
    out = await plan_a.decide(state)
    assert plan_a.clients["main"].complete.await_count == 1 and out["status"] == "unresolved"
    assert not out["terminal_remedy"]["used"]
    expected = {"time": "no_remaining_time", "context": "insufficient_context", "output": "insufficient_output_budget"}
    assert out["terminal_remedy"]["skip_reason"] == expected[limit]


@pytest.mark.asyncio
async def test_missing_usage_is_conservatively_charged_but_remains_unknown(plan_a, store):
    state = parent(plan_a, evidence=True)
    plan_a.clients["main"].complete = AsyncMock(side_effect=[broken(tokens=None), reply(answer(state))])
    out = await plan_a.decide(state)
    assert out["main_usage"][0]["output_budget_charged"] == 31744
    assert out["main_usage"][0]["usage"]["completion_tokens"] is None
    assert export_result({**state, **out}, "test", 1, store=store)["usage"]["output_tokens"] is None


@pytest.mark.asyncio
async def test_request_and_local_prompt_count_use_the_same_thinking_override(plan_a):
    from bcgraph.tokenization import Utf8Counter

    class ModeCounter(Utf8Counter):
        def messages(self, messages, *, enable_thinking=None):
            return super().messages(messages) + {True: 10, False: 20, None: 30}[enable_thinking]

    counter = ModeCounter()
    plan_a.counters["main"] = counter
    state = parent(plan_a, evidence=True)
    plan_a.clients["main"].complete = AsyncMock(side_effect=[broken(), reply(answer(state))])
    await plan_a.decide(state)
    for call in plan_a.clients["main"].complete.await_args_list:
        assert call.kwargs["local_prompt_tokens"] == counter.messages(
            call.args[0], enable_thinking=call.kwargs["enable_thinking"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", [1, 2])
async def test_ambiguous_http_never_adds_or_repeats_a_recovery(plan_a, store, fail_on):
    state = parent(plan_a, evidence=True)
    calls = []

    def backend(request):
        calls.append(request)
        if len(calls) == fail_on:
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(200, json={"choices": [{"message": broken()["assistant"], "finish_reason": "length"}],
                                         "usage": {"completion_tokens": 31744}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        plan_a.clients["main"].http = http
        for _ in range(2):
            out = await plan_a.decide(state)
            assert out["status"] == "unresolved"
        assert len(calls) == fail_on
        result = export_result({**state, **out}, "test", 1, store=store)
        assert result["tool_call_counts"]["main"] == fail_on
        assert result["metadata"]["uncertain_operations"] == 1
        assert result["usage"]["output_tokens"] is None
        assert bool(result["metadata"]["terminal_remedy"].get("used")) is (fail_on == 2)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_after_recovery_commit_replays_without_model_calls(store, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    runtime, backend, http = make_demo_runtime(store)
    enable_plan_a(runtime)
    cfg = {"configurable": {"thread_id": "plan-a-checkpoint"}, "recursion_limit": 256}
    initial = {"query_id": "q", "question": QUESTION, "scope": "plan-a-checkpoint", "attempt": 1,
               "query_deadline": time.time() + 30}
    database = str(tmp_path / "checkpoints.sqlite")

    def missing_body(request):
        raw = backend(request).json()
        content = json.loads(request.content)["messages"][-1]["content"]
        if content.startswith("TASK: DECIDE") and not json.loads(content.rsplit("\n", 1)[1]).get("output_recovery"):
            raw["choices"][0].update(message=broken()["assistant"], finish_reason="length")
            raw["usage"]["completion_tokens"] = 31744
        return httpx.Response(200, json=raw)

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(missing_body)) as main_http:
            runtime.clients["main"].http = main_http
            complete = runtime.clients["main"].complete

            async def crash_after_commit(*args, **kwargs):
                result = await complete(*args, **kwargs)
                if kwargs["operation_id"].endswith(":decision:2"):
                    raise RuntimeError("simulated process loss after recovery commit")
                return result

            runtime.clients["main"].complete = crash_after_commit
            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(runtime, saver)
                with pytest.raises(RuntimeError, match="simulated process loss"):
                    await graph.ainvoke(initial, cfg)
                snapshot = await graph.aget_state(cfg)
                assert snapshot.next == ("decide",)
                assert snapshot.values["query_deadline"] == initial["query_deadline"]
                assert not snapshot.values.get("terminal_remedy")
                failed = export_result({**snapshot.values, "status": "error"}, "test", 1, store=store)
                assert failed["metadata"]["terminal_remedy"]["used"]
                assert failed["tool_call_counts"]["main"] == 3  # plan and two decisions, not the allowance record
    finally:
        await runtime.close()
        await http.aclose()
    restarted, new_backend, new_http = make_demo_runtime(store)
    enable_plan_a(restarted)
    try:
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            result = await build_graph(restarted, saver).ainvoke(None, cfg)
        assert result["status"] == "completed"
        assert result["decision"]["exact_answer"] == "Aurora spectrograph"
        assert result["terminal_remedy"]["used"] and result["decision_round"] == 2
        assert result["final_format_failures"] == 1
        assert result["query_deadline"] == initial["query_deadline"]
        # Restart replays all model work; the only backend call is the final,
        # idempotent chain release introduced after the durable decision.
        assert new_backend.requests == [{'chain_id': 'demo_chain_1'}]
        store.write_result(export_result(result, "test", 1, store=store))
        assert summarize(store.path)["final_output_failure_counts"] == {"empty_content": 1}
    finally:
        await restarted.close()
        await new_http.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cli_resume_uses_original_deadline_without_issuing_requests(plan_a, tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    import yaml
    from bcgraph import cli

    config_path = tmp_path / "config.yaml"
    plan_a.config.workflow.max_query_seconds = 2
    config_path.write_text(yaml.safe_dump(plan_a.config.model_dump()))
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(json.dumps({"query_id": "q", "query": QUESTION}) + "\n")
    args = SimpleNamespace(config=str(config_path), queries=str(query_path), output=str(tmp_path / "batch"),
                           ids=None, ids_file=None, limit=None, resume=False, retry_failed=False)
    now = [100.0]
    monkeypatch.setattr("bcgraph.cli.time.time", lambda: now[0])
    backends = []

    @asynccontextmanager
    async def mock_runtime(config, store):
        runtime, backend, http = make_demo_runtime(store)
        enable_plan_a(runtime)
        runtime.config = config
        backends.append(backend)
        if len(backends) == 1:
            plan = runtime.plan

            async def interrupted(state):
                await plan(state)
                batch_task.cancel()
                await asyncio.sleep(0)

            runtime.plan = interrupted
        try:
            yield runtime
        finally:
            await runtime.close()
            await http.aclose()

    monkeypatch.setattr(cli, "open_runtime", mock_runtime)
    batch_task = asyncio.create_task(cli.run_batch(args))
    with pytest.raises(asyncio.CancelledError):
        await batch_task
    now[0] = 103.0
    args.resume = True
    await cli.run_batch(args)
    assert backends[1].requests == []
    results = list((tmp_path / "batch").glob("run_*.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text())
    assert result["status"] == "timeout"
    assert result["metadata"]["query_deadline"] == 102.0
    assert "Original query deadline" in result["metadata"]["errors"][0]
