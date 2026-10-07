"""These tests execute the REAL LangGraph engine, not the standalone node driver."""
import asyncio
import json
import pytest
pytest.importorskip('langgraph', reason='Real LangGraph dependency is required for graph integration tests')
from bcgraph.graphs import build_graph
from bcgraph.demo import make_demo_runtime, QUESTION
from bcgraph.schemas import FinalDecision

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize('chain,evict,two_cells',[(False,False,False),(True,False,False),
                                             (True,True,False),(True,False,True)])
async def test_real_graph_completion_and_dynamic_fanout(store,chain,evict,two_cells):
    runtime,backend,http=make_demo_runtime(store,chain=chain,evict_once=evict,two_cells=two_cells)
    try:
        output=await build_graph(runtime).ainvoke({'query_id':'demo','question':QUESTION,'scope':'graph','attempt':1},
                                                  {'recursion_limit':256})
        assert output['status']=='completed'
        assert output['decision']['exact_answer']=='Aurora spectrograph'
        assert len(output['main_usage'])==2
        assert len(output['cells'])==(2 if two_cells else 1)
        assert all(c['turn']==2 for c in output['cells'].values())
        assert len(output['evidence'])==(4 if two_cells else 2)
    finally: await runtime.close(); await http.aclose()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_graph_reopens_existing_cell_without_losing_history(store):
    runtime,backend,http=make_demo_runtime(store)
    runtime.config.workflow.max_reader_turns=1
    original_decide=runtime.decide
    async def decide(state):
        if state['reopens']==0:
            decision=FinalDecision(action='research',reopen_cell_id='cell1',
                                   next_queries=['Eira Stone designed instrument'],explanation='Resolve the instrument.')
            job=runtime._reopen_job(state,decision)
            assert job is not None
            return {'next_job':job,'reopens':1,'decision_round':state['decision_round']+1,
                    'decision':decision.model_dump(),'status':'researching'}
        return await original_decide(state)
    runtime.decide=decide
    try:
        result=await build_graph(runtime).ainvoke({'query_id':'d','question':QUESTION,'scope':'reopen','attempt':1},
                                                  {'recursion_limit':256})
        assert result['status']=='completed' and result['reopens']==1
        cell=result['cells']['cell1']
        assert cell['revision']==2 and cell['turn']==2 and len(cell['sources'])==2
        assert cell['metrics'][1]['request_mode']=='chain_delta'
    finally: await runtime.close(); await http.aclose()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlite_checkpoint_and_completed_graph_state(store,tmp_path):
    module=pytest.importorskip('langgraph.checkpoint.sqlite.aio')
    runtime,backend,http=make_demo_runtime(store,two_cells=True)
    cfg={'configurable':{'thread_id':'test-thread'},'recursion_limit':256}
    try:
        async with module.AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
            graph=build_graph(runtime,saver)
            output=await graph.ainvoke({'query_id':'d','question':QUESTION,'scope':'checkpoint','attempt':1},cfg)
            snapshot=await graph.aget_state(cfg)
            assert not snapshot.next and snapshot.values['final_text']==output['final_text']
        async with module.AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as saver:
            restored=await build_graph(runtime,saver).aget_state(cfg)
            assert restored.values['status']=='completed' and len(restored.values['cells'])==2
    finally: await runtime.close(); await http.aclose()
