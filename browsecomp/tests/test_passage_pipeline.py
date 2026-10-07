import json
from copy import deepcopy
import pytest
from bcgraph.passage_demo import make_demo_runtime, QUESTION
from bcgraph.app import export_result

async def drive(runtime, scope='new-pipeline'):
    """Real nodes + real ChatClient, NOT a substitute for LangGraph integration."""
    state={'query_id':'synthetic','question':QUESTION,'scope':scope,'attempt':1}
    state.update(await runtime.plan(state))
    jobs=[(j,None) for j in state['jobs']]
    for stage in range(3):
        for job,previous in jobs:
            cell=runtime.init_cell({**state,'job':job,'previous':previous})
            for _ in range(20):
                cell.update(await runtime.cell_fetch(cell))
                if cell['stop_reason']: break
                cell.update(await runtime.cell_read(cell))
                if cell['stop_reason']: break
                cell.update(runtime.cell_validate(cell))
                if cell['stop_reason']: break
            else: raise AssertionError('unbounded research loop')
            cell.pop('packed',None); cell.pop('reply',None)
            state['cells'][cell['cell_id']]=cell
        state.update(runtime.collect(state)); state.update(await runtime.decide(state))
        if not state.get('next_job'):
            state.update(runtime.render(state)); return state
        job=state['next_job']; jobs=[(job,state['cells'][job['id']])]
    raise AssertionError('unbounded reopen loop')

@pytest.mark.parametrize('chain,evict,two',[(False,False,False),(True,False,False),(True,True,False),(True,False,True)])
async def test_active_pipeline_raw_history_chain_and_export(store,chain,evict,two):
    rt,http_fixture,http=make_demo_runtime(store,chain=chain,evict_once=evict,two_cells=two)
    try:
        s=await drive(rt)
        assert s['status']=='completed'
        assert s['decision']['exact_answer']=='Aurora spectrograph'
        assert s['decision']['semantic_correctness_verified'] is False
        assert s['evidence']=={} and len(s['main_usage'])==2
        assert len(s['cells'])==(2 if two else 1)
        assert all(c['turn']==2 for c in s['cells'].values())
        c=s['cells']['cell1']
        assert all('quote' not in json.loads(m['content']) for m in c['raw_history'] if m['role']=='assistant')
        if chain and not evict: assert c['metrics'][1]['request_mode']=='chain_delta'
        result=export_result(s,'config',1000,store=store)
        assert result['status']=='completed' and result['metadata']['selected_passage_count']>=2
        assert result['metadata']['accepted_evidence_count']==0
        assert '[demo_instrument]' in result['result'][0]['output']
        assert result['metadata']['semantic_correctness_verified'] is False
        assert result['tool_call_counts']['main']==2
    finally: await rt.close(); await http.aclose()

async def test_unknown_final_id_gets_one_repair_not_discarded_answer(store):
    rt,b,http=make_demo_runtime(store)
    attempts=[]
    def mutate(obj,messages,reader,body):
        if not reader and obj.get('action')=='answer':
            attempts.append(deepcopy(obj))
            if len(attempts)==1: return {**obj,'citations':['P999']}
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='completed' and s['final_repair_count']==1
        assert len(attempts)==2 and len(s['main_usage'])==3
        assert 'final_validation_failed' in s['delivery_issues'][0]
        assert not any(e.startswith('final_validation_failed') for e in s['errors'])
        assert export_result(s,'c',1)['metadata']['execution_status']=='degraded'
    finally: await rt.close(); await http.aclose()

async def test_missing_fact_confidence_extra_metadata_do_not_fail_delivery(store):
    rt,b,http=make_demo_runtime(store)
    def mutate(obj,messages,reader,body):
        if not reader and obj.get('action')=='answer':
            return {**obj,'missing_fact':'one auxiliary fact is not fully confirmed','confidence':0.7,'extra_note':'x'}
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='completed' and s['final_repair_count']==0
        assert export_result(s,'c',1)['metadata']['execution_status']=='ok'
        assert 'Confidence:' not in s['final_text']
    finally: await rt.close(); await http.aclose()

async def test_invalid_reader_ids_do_not_blind_main_to_actual_originals(store):
    rt,b,http=make_demo_runtime(store)
    def mutate(obj,messages,reader,body):
        if reader: return {**obj,'selected_passage_ids':['P999']}
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='completed'
        assert all(not c['selected_passages'] for c in s['cells'].values())
        assert any('unknown_passage_id' in e for c in s['cells'].values() for e in c['protocol_issues'])
        assert s['decision']['citation_docids']==['demo_identity','demo_instrument']
    finally: await rt.close(); await http.aclose()

async def test_abstention_reopens_old_chain_within_global_budget(store):
    rt,b,http=make_demo_runtime(store)
    rt.config.workflow.max_reader_turns=1
    try:
        s=await drive(rt)
        c=s['cells']['cell1']
        assert s['status']=='completed' and s['reopens']==1 and c['revision']==2
        assert c['metrics'][1]['request_mode']=='chain_delta'
        assert c['searches_used']<=rt.config.workflow.max_searches_per_query
        assert len(c['raw_history'])==5  # system + two real user/assistant turns
        assert len(s['main_usage'])==3
    finally: await rt.close(); await http.aclose()

async def test_unrepairable_final_never_becomes_completed_or_loops(store):
    rt,b,http=make_demo_runtime(store)
    rt.config.workflow.max_reopens=0
    def mutate(obj,messages,reader,body):
        if not reader and 'action' in obj: return '{"action":"answer","exact_answer":'
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='unresolved' and s['decision']['exact_answer']==''
        assert len(s['main_usage'])==3 and s['final_repair_count']==1
        result=export_result(s,'c',1)
        assert result['metadata']['execution_status']=='error'
        assert result['metadata']['outcome']=='failed'
        assert s['decision_exit_cause']=='delivery_failed_after_bounded_recovery'
    finally: await rt.close(); await http.aclose()

async def test_truncated_final_recovery_is_compact_and_bounded(store):
    rt,b,http=make_demo_runtime(store)
    final_attempts=[]
    def mutate(obj,messages,reader,body):
        if not reader and 'action' in obj:
            final_attempts.append((deepcopy(messages),body['max_tokens']))
            if len(final_attempts)==1:
                return '{"action":"answer","exact_answer":"Aurora spectrograph","citations":["D1P1"],"explanation":"loop loop'
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='completed' and s['final_repair_count']==1
        assert [cap for _,cap in final_attempts]==[rt.config.workflow.final_output_tokens,1024]
        feedback=final_attempts[1][0][-1]['content']
        assert 'at most two short sentences' in feedback
        assert 'do not repeat any sentence or phrase' in feedback
    finally: await rt.close(); await http.aclose()

async def test_returned_answer_is_not_inferred_by_python_from_candidate_hint(store):
    rt,b,http=make_demo_runtime(store)
    rt.config.workflow.max_reopens=0
    def mutate(obj,messages,reader,body):
        if not reader and 'action' in obj: return {'action':'unresolved','missing_fact':'A required relation is explicitly contradicted.'}
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='unresolved' and not s['decision']['exact_answer']
        assert s['final_review_count']==1 and len(s['main_usage'])==3
    finally: await rt.close(); await http.aclose()

async def test_context_exhaustion_does_not_spend_more_searches(store):
    rt,b,http=make_demo_runtime(store)
    try:
        s={'query_id':'x','question':QUESTION,'scope':'ctx','attempt':1}
        s.update(await rt.plan(s)); c=rt.init_cell({**s,'job':s['jobs'][0],'previous':None})
        c['turn']=1;c['backend']='reader'; c['raw_history']=[{'role':'system','content':'x'*100000}]
        before=len(store.operations('ctx'))
        update=await rt.cell_fetch(c)
        assert update['stop_reason']=='context_budget_exhausted'
        assert len(store.operations('ctx'))==before
    finally: await rt.close(); await http.aclose()


def test_unread_offsets_do_not_reuse_stale_source_tail():
    from bcgraph.evidence import source_record
    from bcgraph.passage_runtime import PassageRuntime
    text='x'*100
    first=source_record('doc',text,0,60)
    last=source_record('doc',text,60,100)
    cell={'sources':{first['source_id']:first,last['source_id']:last},'doc_catalog':{'doc':{}}}
    assert PassageRuntime._unread_document_requests(cell)==[]
    cell['sources']={last['source_id']:last}
    assert PassageRuntime._unread_document_requests(cell)==[{'docid':'doc','offset':0}]

async def test_out_of_range_reader_offset_does_not_trigger_useless_fetch(store):
    rt,b,http=make_demo_runtime(store)
    def mutate(obj,messages,reader,body):
        if reader:return {**obj,'read_more':[{'docid':'demo_identity','offset':999999}]}
    b.mutate=mutate
    try:
        s=await drive(rt)
        assert s['status']=='completed'
        assert any('invalid_read_request' in e for e in s['cells']['cell1']['protocol_issues'])
        assert s['cells']['cell1']['turn']==2
    finally: await rt.close();await http.aclose()
