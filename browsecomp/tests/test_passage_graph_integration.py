"""Required release gate: these execute the real LangGraph, not the node driver."""
import pytest
pytest.importorskip('langgraph')
from bcgraph.graphs import build_graph
from bcgraph.passage_demo import make_demo_runtime, QUESTION

@pytest.mark.integration
@pytest.mark.parametrize('chain,evict,two',[(False,False,False),(True,False,False),(True,True,False),(True,False,True)])
async def test_actual_passage_graph(store,chain,evict,two):
    rt,b,http=make_demo_runtime(store,chain=chain,evict_once=evict,two_cells=two)
    try:
        s=await build_graph(rt).ainvoke({'query_id':'x','question':QUESTION,'scope':'real','attempt':1},
                                      {'recursion_limit':256})
        assert s['protocol']=='passages-v2' and s['status']=='completed'
        assert all(c['selected_passages'] and c['protocol']=='passages-v2' for c in s['cells'].values())
        assert s['decision']['exact_answer']=='Aurora spectrograph'
    finally: await rt.close(); await http.aclose()

@pytest.mark.integration
async def test_actual_sqlite_reopen_recovery_and_state_channels(store,tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    rt,b,http=make_demo_runtime(store)
    rt.config.workflow.max_reader_turns=1
    cfg={'configurable':{'thread_id':'passage-thread'},'recursion_limit':256}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as cp:
            graph=build_graph(rt,cp)
            s=await graph.ainvoke({'query_id':'x','question':QUESTION,'scope':'realcp','attempt':1},cfg)
            assert s['status']=='completed' and s['reopens']==1
            assert s['cells']['cell1']['metrics'][1]['request_mode']=='chain_delta'
            snap=await graph.aget_state(cfg)
            assert not snap.next and snap.values['protocol']=='passages-v2'
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'checkpoints.sqlite')) as cp:
            snap=await build_graph(rt,cp).aget_state(cfg)
            assert snap.values['decision']==s['decision']
            assert snap.values['cells']['cell1']['selected_passages']==s['cells']['cell1']['selected_passages']
    finally: await rt.close(); await http.aclose()
