"""Small synthetic reproductions of the observed smoke failures (no gold answers)."""
from copy import deepcopy
import json
import pytest
from bcgraph.schemas import FinalDecision, ReaderReply, parse_reader_output, parse_json_object
from bcgraph.evidence import (source_record, source_aliases, evidence_aliases, accept_reply,
                              exact_span, render_answer)
from bcgraph.prompts import reader_message, final_message, planner_message
from bcgraph.demo import make_demo_runtime, drive_nodes_for_test, QUESTION
from bcgraph.app import export_result

C = {'c1': {'id':'c1','description':'Director in 2007','time_scope':'2007','required':True,'answer_target':False},
     'target': {'id':'target','description':'Instrument','time_scope':'unspecified','required':True,'answer_target':True}}
TEXT = 'Eira was director in 2007.\n\nEira designed Aurora.'


def fixture():
    s = source_record('doc', TEXT, 0, len(TEXT))
    return s, {'candidate':'Eira', 'constraint_id':'target', 'source_id':'S1',
               'quote':'Eira designed Aurora.', 'answer_value':'Aurora'}


@pytest.mark.parametrize('confidence', [0.7, 0.9, 1, 90, '90%', None, {'unused':True}])
def test_confidence_is_not_part_of_the_final_contract(confidence):
    d = FinalDecision.model_validate({'action':'research', 'confidence':confidence,
                                      'candidate':None, 'exact_answer':None, 'reopen_cell_id':'cell1'})
    assert d.candidate == d.exact_answer == ''
    assert 'confidence' not in d.model_dump()


def test_answer_cannot_be_manufactured_from_null():
    with pytest.raises(ValueError, match='nonempty'):
        FinalDecision.model_validate({'action':'answer','exact_answer':None})
    with pytest.raises(ValueError):
        FinalDecision.model_validate({'action':'answer','exact_answer':['guess']})


def test_optional_reader_advisory_fields_are_not_required():
    p = ReaderReply.model_validate({'evidence':None, 'next_queries':None, 'complete':'unused',
                                   'confidence':0.9, 'summary':None})
    assert p.evidence == [] and p.summary == ''
    assert 'complete' not in p.model_dump()


def test_short_source_id_and_server_supplied_time_scope():
    s,p = fixture()
    ledger,errors,_=accept_reply(ReaderReply(evidence=[p]),{s['source_id']:s},C,{},'cell1',1)
    assert not errors
    row=next(iter(ledger.values()))
    assert row['source_id']==s['source_id'] and row['time_scope']=='unspecified'
    assert row['claim']==row['quote']


def test_aliases_are_append_only_and_unknown_alias_not_guessed():
    s,p=fixture(); sources={s['source_id']:s}
    t=source_record('second','Other source.',0,13)
    assert source_aliases(sources)[s['source_id']]=='S1'
    assert source_aliases({**sources,t['source_id']:t})[s['source_id']]=='S1'
    p['source_id']='S999'
    ledger,errors,_=accept_reply(ReaderReply(evidence=[p]),sources,C,{},'cell1',1)
    assert not ledger and 'Source was not actually shown' in errors[0]


def test_short_retraction_id_still_checks_ownership():
    s,p=fixture();sources={s['source_id']:s}
    ledger,_,_=accept_reply(ReaderReply(evidence=[p]),sources,C,{},'cell1',1)
    other,errors,_=accept_reply(ReaderReply(retract_ids=['E1']),sources,C,ledger,'other',2)
    assert errors and next(iter(other.values()))['active']
    own,errors,_=accept_reply(ReaderReply(retract_ids=['E1']),sources,C,ledger,'cell1',2)
    assert not errors and not next(iter(own.values()))['active']


def test_whitespace_only_match_preserves_original_offsets_and_text():
    s,p=fixture();p.update(quote='Eira was director in 2007. Eira designed Aurora.')
    ledger,errors,_=accept_reply(ReaderReply(evidence=[p]),{s['source_id']:s},C,{},'cell1',1)
    assert not errors
    e=next(iter(ledger.values()))
    assert e['quote']==TEXT and e['quote_match']=='whitespace_only'
    assert TEXT[e['quote_start']:e['quote_end']]==e['quote']


@pytest.mark.parametrize('quote', ['Eira was director in 2009.', 'Eira ... designed Aurora.',
                                   'Eira was director. Eira designed Aurora.', 'Eira designed Orion.'])
def test_whitespace_matching_does_not_repair_facts_or_join_fragments(quote):
    assert exact_span(TEXT,quote) is None


def test_truncated_reader_keeps_only_complete_source_checkable_objects():
    s,p=fixture()
    text='```json\n{"evidence":['+json.dumps(p)+',{"candidate":"unterminated'
    reply,partial=parse_reader_output(text,'length')
    assert partial and reply.evidence==[p]
    assert not reply.next_queries and not reply.retract_ids
    ledger,errors,_=accept_reply(reply,{s['source_id']:s},C,{},'cell1',1)
    assert len(ledger)==1 and not errors


@pytest.mark.parametrize('text,finish', [
    ('{"evidence":[{"source_id":"S11111111111111','length'),
    ('{"evidence":[{"quote":"broken','stop'),
    ('{"evidence":[]} {"evidence":[]}', 'length'),
    ('{"evidence":[],"unknown_field":1}', 'length'),
    ('narrative {"evidence":[{"quote":"broken', 'length'),
])
def test_no_salvage_of_ambiguous_or_fabricated_roots(text,finish):
    with pytest.raises(ValueError):parse_reader_output(text,finish)


def state_for_prompt():
    s,p=fixture()
    state={'turn':1,'question':'Which instrument?','focus':'Verify identity then instrument',
           'constraints':C,'constraint_ids':list(C),'sources':{s['source_id']:s},'evidence':{},
           'search_limit':8,'searches_used':3}
    return state,s


def test_historical_aliases_and_small_state_are_visible_not_hashes():
    state,s=state_for_prompt()
    text=reader_message(state,{})['content']; d=json.loads(text)
    assert d['historical_sources'][0]['source_id']=='S1'
    assert s['source_id'] not in text and s['document_sha256'] not in text
    assert d['question']==state['question'] and d['max_evidence_items']==4


def test_main_does_not_treat_unverified_summary_or_claim_as_evidence():
    s,p=fixture();ledger,_,_=accept_reply(ReaderReply(evidence=[p]),{s['source_id']:s},C,{},'cell1',1)
    row=next(iter(ledger.values())); row['claim']='FABRICATED CLAIM'
    msg=final_message('Q',C,[row],{}, {'cell1':{'focus':'F','summary':'FABRICATED SUMMARY'}},False)['content']
    assert 'FABRICATED CLAIM' not in msg and 'FABRICATED SUMMARY' not in msg
    d=json.loads(msg.rsplit('\n',1)[-1])
    assert d['validated_evidence'][0]['evidence_id']=='E1'
    assert row['evidence_id'] not in msg and not d['can_reopen']
    assert '"action":"research"' not in msg


def test_planner_example_has_multiple_constraints_and_no_placeholder_searches():
    text=planner_message('A multi-clue question.',1)['content']
    assert '"id":"c1"' in text and '"id":"c2"' in text and '"id":"target"' in text
    assert 'NOT real names' in text


@pytest.mark.asyncio
async def test_target_only_plan_does_not_auto_stop_on_one_name(store):
    runtime,_,http=make_demo_runtime(store)
    try:
        s,p=fixture()
        state={'scope':'test','cell_id':'cell1','turn':1,'constraints':{'target':C['target']},
               'constraint_ids':['target'],'sources':{s['source_id']:s},'evidence':{},
               'reply':{'assistant':{'content':json.dumps({'evidence':[p],'next_queries':['verify actual biographical clue']})},'finish_reason':'stop'},
               'pending_queries':[],'seen_queries':[],'summary':'','no_progress_turns':0,
               'turn_limit':3,'output_limit':5000,'output_used':200,'documents_used':1,
               'document_limit':20,'searches_used':1,'search_limit':8}
        out=runtime.cell_validate(state)
        assert out['stop_reason'] != 'evidence_sufficient'
        assert out['pending_queries']
    finally:await runtime.close();await http.aclose()


@pytest.mark.asyncio
async def test_protocol_patch_still_runs_dense_and_linear_chain_nodes(store):
    runtime,backend,http=make_demo_runtime(store,chain=True,evict_once=True)
    try:
        state=await drive_nodes_for_test(runtime,{'query_id':'demo','question':QUESTION,'scope':'alias','attempt':1})
        assert state['status']=='completed'
        assert 'Confidence:' not in state['final_text']
        exp=export_result(state,'test',100)
        assert exp['metadata']['execution_status']=='ok' and exp['metadata']['outcome']=='answered'
        assert exp['metadata']['shown_docids']==['demo_identity','demo_instrument']
        assert exp['metadata']['accepted_evidence_count']==2
        assert any(b.get('chain_append_start')==1 for b in backend.requests)
    finally:await runtime.close();await http.aclose()


def test_abstention_does_not_become_answer_completion_or_disappear_from_denominator():
    out=export_result({'query_id':'q','status':'unresolved','decision':{'action':'unresolved'}},'hash',1)
    assert out['status']=='unresolved'
    assert out['metadata']['execution_status']=='ok'
    assert out['metadata']['outcome']=='abstained'
    out=export_result({'query_id':'q','status':'unresolved','errors':['final_format_error: bad json']},'hash',1)
    assert out['metadata']['execution_status']=='error'


def test_render_real_docids_not_model_evidence_aliases_or_hash_digits():
    text=render_answer({'exact_answer':'Aurora','explanation':'Supported by [E1]. Reject [E999] and [e_abc123].',
                       'citation_aliases':{'E1':'56789'},'citation_docids':['56789']})
    assert '[56789]' in text
    assert '[E1]' not in text and '[E999]' not in text and '[e_abc123]' not in text
    assert 'Confidence' not in text


def test_final_protocol_error_is_not_successful_abstention():
    out=export_result({'query_id':'q','status':'unresolved','errors':['final_format_error: bad json']},'hash',1)
    assert out['status']=='unresolved'  # Keep legacy evaluator denominator contract.
    assert out['metadata']['execution_status']=='error'
    assert out['metadata']['outcome']=='failed'


@pytest.mark.parametrize("field,value", [("source_id", []), ("source_id", {}), ("constraint_id", [])])
def test_malformed_identifiers_are_rejected_per_entry_not_python_type_crashes(field,value):
    s,p=fixture();p[field]=value
    ledger,errors,_=accept_reply(ReaderReply(evidence=[p]),{s["source_id"]:s},C,{},"cell1",1)
    assert not ledger and errors
