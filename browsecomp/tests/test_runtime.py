import asyncio
from copy import deepcopy
import json
import pytest
from bcgraph.demo import make_demo_runtime, drive_nodes_for_test, QUESTION
from bcgraph.runtime import unique_queries, distribute
from bcgraph.packer import pack_documents
from bcgraph.config import WorkflowConfig
from bcgraph.tokenization import Utf8Counter
from bcgraph.prompts import READER_SYSTEM
from bcgraph.evidence import source_record
from bcgraph.app import export_result
from bcgraph.admission import PriorityGate
from bcgraph.metrics import summarize

@pytest.mark.asyncio
@pytest.mark.parametrize('chain,evict',[(False,False),(True,False),(True,True)])
async def test_core_end_to_end_nodes(store,chain,evict):
    runtime,backend,http=make_demo_runtime(store,chain=chain,evict_once=evict)
    try:
        result=await drive_nodes_for_test(runtime,{'query_id':'demo','question':QUESTION,'scope':'demo','attempt':1})
        assert result['status']=='completed' and result['decision']['exact_answer']=='Aurora spectrograph'
        assert len(result['main_usage'])==2 and result['cells']['cell1']['turn']==2
        assert result['cells']['cell1']['stop_reason']=='evidence_sufficient'
        cell=result['cells']['cell1']
        assert len(cell['sources'])==2 and len(cell['evidence'])==2
        assert cell['raw_history'][2]['content'].endswith('\n')
        assert len(backend.requests)==(5 if evict else 4)
        store.write_result(export_result(result,'test',10))
        assert summarize(store.path)['accuracy'] is None
    finally:
        await runtime.close(); await http.aclose()

@pytest.mark.asyncio
async def test_whole_node_replay_does_not_repeat_http_or_tools(store):
    runtime,backend,http=make_demo_runtime(store)
    state={'query_id':'demo','question':QUESTION,'scope':'demo','attempt':1}
    try:
        first=await drive_nodes_for_test(runtime,state)
        count=len(backend.requests)
        second=await drive_nodes_for_test(runtime,state)
        assert len(backend.requests)==count
        assert first['decision']==second['decision']
    finally: await runtime.close(); await http.aclose()

@pytest.mark.asyncio
async def test_budget_stops_without_forced_three_turns(store):
    runtime,backend,http=make_demo_runtime(store)
    runtime.config.workflow.max_reader_turns=1
    try:
        result=await drive_nodes_for_test(runtime,{'query_id':'demo','question':QUESTION,'scope':'demo','attempt':1})
        assert result['cells']['cell1']['turn']==1
        assert result['status']=='unresolved'
    finally: await runtime.close(); await http.aclose()

@pytest.mark.asyncio
async def test_two_cell_budget_allocation_respects_global_cap(store):
    runtime,backend,http=make_demo_runtime(store,two_cells=True)
    try:
        result=await drive_nodes_for_test(runtime,{'query_id':'demo','question':QUESTION,'scope':'demo','attempt':1})
        cfg=runtime.config.workflow
        assert len(result['cells'])==2
        assert sum(c['searches_used'] for c in result['cells'].values())<=cfg.max_searches_per_query
        assert sum(c['documents_used'] for c in result['cells'].values())<=cfg.max_document_fetches_per_query
        assert sum(c['output_used'] for c in result['cells'].values())<=cfg.max_total_reader_output_tokens
    finally: await runtime.close(); await http.aclose()

def packed_state():
    return {'turn':0,'question':QUESTION,'focus':QUESTION,'raw_history':[{'role':'system','content':READER_SYSTEM}],
            'sources':{},'constraints':{'x':{'description':QUESTION,'time_scope':'unspecified'}},'constraint_ids':['x'],
            'evidence':{},'validation_errors':[],'doc_catalog':{},'cell_id':'c'}

def test_packer_bounds_full_history_not_only_new_documents():
    state=packed_state(); cfg=WorkflowConfig(first_source_tokens=2000,context_reserve_tokens=64)
    counter=Utf8Counter(); documents=[{'docid':'long','text':('Eira designed Aurora.\n\n'*1000)}]
    packed=pack_documents(state,documents,counter,cfg,8000,256)
    assert packed['message'] is not None
    assert counter.messages([*state['raw_history'],packed['message']])+256+64<=8000
    assert packed['source_tokens']<=2000
    state['raw_history'].append({'role':'assistant','content':'z'*20000})
    out=pack_documents(state,documents,counter,cfg,8000,256)
    assert out['stop_reason']=='context_budget_exhausted'


@pytest.mark.asyncio
async def test_background_packing_preserves_exact_output_and_inputs(store):
    runtime, _, http = make_demo_runtime(store)
    state = packed_state()
    documents = [{'docid': 'long', 'text': 'Eira designed Aurora.\n\n' * 1000}]
    before = deepcopy((state, documents))
    expected = pack_documents(state, documents, runtime.counters['reader'],
                              runtime.config.workflow,
                              runtime.clients['reader'].config.max_context_tokens, 256)
    try:
        assert await runtime._pack_documents(state, documents, 'reader', 256) == expected
        assert (state, documents) == before
    finally:
        await runtime.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_background_packing_leaves_event_loop_responsive_and_isolates_counter(store, monkeypatch):
    import threading
    runtime, _, http = make_demo_runtime(store)
    release = threading.Event()
    main_thread = threading.get_ident()
    original = runtime.counters['reader']

    def blocked_pack(state, documents, counter, *args):
        assert threading.get_ident() != main_thread
        assert counter is not original
        assert release.wait(2), 'The event loop could not release the worker'
        raise ValueError('worker failure')

    monkeypatch.setattr('bcgraph.runtime.pack_documents', blocked_pack)
    handle = asyncio.get_running_loop().call_later(.05, release.set)
    try:
        with pytest.raises(ValueError, match='worker failure'):
            await runtime._pack_documents({}, [], 'reader', 256)
    finally:
        handle.cancel()
        release.set()
        await runtime.close()
        await http.aclose()

def test_previously_seen_document_is_not_repeated_without_request():
    state=packed_state(); text='A useful but short original document.'
    src=source_record('d',text,0,len(text)); state['sources']={src['source_id']:src}
    out=pack_documents(state,[{'docid':'d','text':text}],Utf8Counter(),WorkflowConfig(),20000,256)
    assert out['stop_reason']=='no_novel_evidence'
    out=pack_documents(state,[{'docid':'d','text':text,'requested_offset':0}],Utf8Counter(),WorkflowConfig(),20000,256)
    assert out['message'] is not None

def test_dedup_and_zero_budget():
    assert unique_queries([' A  B ','a b','C'],[],5)==['A B','C']
    assert unique_queries(['a'],[],0)==[]
    assert distribute(10,3)==[4,3,3]

@pytest.mark.asyncio
async def test_priority_gate_cancellation_does_not_leak():
    gate=PriorityGate(1)
    async with gate.slot():
        async def blocked():
            async with gate.slot(): pass
        task=asyncio.create_task(blocked()); await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert gate.snapshot()['active']==1 and gate.snapshot()['pending']==0
    async with gate.slot(): assert gate.active==1
    assert gate.active==0

@pytest.mark.asyncio
async def test_queued_continuation_has_priority():
    gate=PriorityGate(1); order=[]
    async def worker(name,continuation):
        async with gate.slot(continuation=continuation): order.append(name)
    async with gate.slot():
        cold=asyncio.create_task(worker('cold',False)); await asyncio.sleep(0)
        warm=asyncio.create_task(worker('warm',True)); await asyncio.sleep(0)
    await asyncio.gather(cold,warm)
    assert order==['warm','cold']
