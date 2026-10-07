import asyncio

import pytest
pytest.importorskip("langgraph", reason="Real LangGraph/checkpoint dependency required")
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bcgraph.app import export_result
from bcgraph.demo import make_demo_runtime, QUESTION
from bcgraph.graphs import build_graph


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mid_cell_restart_replays_committed_calls_and_cold_starts_new_process(store, tmp_path):
    config = {"configurable": {"thread_id": "restart"}, "recursion_limit": 256}
    state = {"query_id": "q", "question": QUESTION, "scope": "restart:q", "attempt": 1}
    runtime, backend, http = make_demo_runtime(store)
    original = runtime.cell_read

    async def crash_after_commit(cell):
        await original(cell)
        raise RuntimeError("simulated process loss after HTTP commit")

    runtime.cell_read = crash_after_commit
    database = str(tmp_path / "checkpoints.sqlite")
    try:
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(runtime, saver)
            with pytest.raises(RuntimeError, match="simulated process loss"):
                await graph.ainvoke(state, config)
            snapshot = await graph.aget_state(config)
            assert snapshot.next
            failed = export_result({**snapshot.values, "status": "error"}, "config", 100, store=store)
            assert failed["tool_call_counts"]["reader"] == 1
            assert failed["retrieved_docids"] == ["demo_identity"]
            assert len(backend.requests) == 2
    finally:
        await runtime.close()
        await http.aclose()

    restarted, new_backend, new_http = make_demo_runtime(store)
    try:
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            result = await build_graph(restarted, saver).ainvoke(None, config)
        assert result["status"] == "completed"
        # Cache release is a control request, not another model completion.
        assert len([r for r in new_backend.requests if 'messages' in r]) == 2
        assert result["cells"]["cell1"]["metrics"][0]["journal_replay"]
        assert result["cells"]["cell1"]["metrics"][1]["request_mode"] == "cold_stale_handle"
        assert "chain_id" not in new_backend.requests[0]
        messages = new_backend.requests[0]["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
        assert messages[2] == result["cells"]["cell1"]["raw_history"][2]
    finally:
        await restarted.close()
        await new_http.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_cell_can_reach_second_turn_before_slow_cell_finishes_first(store):
    runtime, _, http = make_demo_runtime(store, two_cells=True)
    fast_second_turn = asyncio.Event()
    original = runtime.cell_read

    async def staggered_read(cell):
        if cell["cell_id"] == "cell2" and cell["turn"] == 0:
            await asyncio.wait_for(fast_second_turn.wait(), timeout=5)
        if cell["cell_id"] == "cell1" and cell["turn"] == 1:
            fast_second_turn.set()
        return await original(cell)

    runtime.cell_read = staggered_read
    try:
        result = await build_graph(runtime).ainvoke(
            {"query_id": "q", "question": QUESTION, "scope": "stagger", "attempt": 1},
            {"recursion_limit": 256})
        assert result["status"] == "completed"
        assert fast_second_turn.is_set()
        assert len(result["main_usage"]) == 2
    finally:
        await runtime.close()
        await http.aclose()
