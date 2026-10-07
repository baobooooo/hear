from copy import deepcopy
import pytest
from bcgraph.evidence import (source_record, merge_sources, merge_cells, accept_reply,
                              candidate_status, validate_final, render_answer)
from bcgraph.schemas import ReaderReply, parse_json_object

CONSTRAINTS = {
    'identity': {'id':'identity','description':'Director','time_scope':'2007','required':True,'answer_target':False},
    'target': {'id':'target','description':'Instrument','time_scope':'unspecified','required':True,'answer_target':True}}
TEXT = 'Eira was director in 2007. Eira designed Aurora.'

def setup_evidence():
    src = source_record('doc', TEXT, 0, len(TEXT))
    def proposal(cid, quote, answer=None, candidate='Eira'):
        out = dict(candidate=candidate, constraint_id=cid, time_scope=CONSTRAINTS[cid]['time_scope'],
                   relation='SUPPORTS', source_id=src['source_id'], quote=quote, claim=quote)
        if answer: out['answer_value'] = answer
        return out
    items = [proposal('identity', 'Eira was director in 2007.'), proposal('target', 'Eira designed Aurora.', 'Aurora')]
    return src, items

def accept(items, src, previous=None, turn=1):
    return accept_reply(ReaderReply(evidence=items), {src['source_id']:src}, CONSTRAINTS, previous or {}, 'c1', turn)

def test_full_candidate_is_ready():
    src, items = setup_evidence(); ledger, errors, changes = accept(items, src)
    assert not errors and changes == 2
    assert candidate_status(ledger, CONSTRAINTS)['eira']['ready']

def test_two_candidates_cannot_complete_each_other():
    src, items = setup_evidence(); items[1]['candidate']='Other person'
    ledger, _, _ = accept(items, src)
    assert not any(v['ready'] for v in candidate_status(ledger, CONSTRAINTS).values())

@pytest.mark.parametrize('change', [{'quote':'fabricated'}, {'source_id':'never_shown'}, {'time_scope':'2010'}, {'candidate':'unknown'}, {'constraint_id':'absent'}])
def test_invalid_evidence_is_rejected_individually(change):
    src, items = setup_evidence(); items[1].update(change)
    ledger, errors, count = accept(items, src)
    assert len(ledger)==1 and count==1 and len(errors)==1

def test_old_source_can_support_later_turn():
    src, items = setup_evidence(); old,_,_=accept(items[:1],src)
    ledger, errors, count = accept(items[1:],src,old,2)
    assert not errors and count==1 and len(ledger)==2

def test_duplicate_evidence_is_idempotent():
    src, items=setup_evidence(); ledger,_,_=accept(items,src)
    again,errors,count=accept(items,src,ledger,2)
    assert again==ledger and not errors and count==0

def test_retracted_evidence_cannot_reactivate_by_repetition():
    src,items=setup_evidence(); ledger,_,_=accept(items,src); ident=next(iter(ledger))
    revised,errors,_=accept_reply(ReaderReply(evidence=items,retract_ids=[ident]),{src['source_id']:src},CONSTRAINTS,ledger,'c1',2)
    assert not errors and not revised[ident]['active'] and revised[ident]['revision']==2

def test_unowned_retraction_is_rejected():
    src,items=setup_evidence(); ledger,_,_=accept(items,src)
    _,errors,_=accept_reply(ReaderReply(retract_ids=list(ledger)),{src['source_id']:src},CONSTRAINTS,ledger,'other',2)
    assert len(errors)==2

def test_final_requires_cited_target_value():
    src,items=setup_evidence(); ledger,_,_=accept(items,src)
    decision=dict(candidate='Eira',exact_answer='Aurora',evidence_ids=list(ledger))
    assert validate_final(decision,ledger,CONSTRAINTS)==(True,'complete')
    decision['exact_answer']='Other'
    assert not validate_final(decision,ledger,CONSTRAINTS)[0]

def test_required_contradiction_blocks_answer():
    src,items=setup_evidence(); other={**items[0],'relation':'CONTRADICTS'}
    ledger,_,_=accept([*items,other],src)
    assert not validate_final(dict(candidate='Eira',exact_answer='Aurora',evidence_ids=list(ledger)),ledger,CONSTRAINTS)[0]

def test_conflicting_target_values_block_answer():
    src,items=setup_evidence(); other={**items[1],'answer_value':'Orion'}
    ledger,_,_=accept([*items,other],src)
    assert not validate_final(dict(candidate='Eira',exact_answer='Aurora',evidence_ids=list(ledger)),ledger,CONSTRAINTS)[0]

def test_strict_coverage_is_optional_but_labeled():
    src,items=setup_evidence(); ledger,_,_=accept(items[1:],src)
    decision=dict(candidate='Eira',exact_answer='Aurora',evidence_ids=list(ledger))
    assert validate_final(decision,ledger,CONSTRAINTS)==(True,'answer_with_gaps')
    assert not validate_final(decision,ledger,CONSTRAINTS,True)[0]

def test_source_version_and_immutable_collision():
    a=source_record('d','abcdef',0,3); b=source_record('d','abcxef',0,3)
    assert a['source_id']!=b['source_id']
    assert len(merge_sources({a['source_id']:a},{b['source_id']:b}))==2
    with pytest.raises(ValueError): merge_sources({a['source_id']:a},{a['source_id']:{**a,'text':'bad'}})

def test_cell_snapshot_reducer_handles_replay_and_reopen():
    one={'cell_id':'c','revision':1,'value':'old'}; two={**one,'revision':2,'value':'new'}
    assert merge_cells({'c':one},{'c':one})=={'c':one}
    assert merge_cells({'c':two},{'c':one})=={'c':two}
    assert merge_cells({'c':one},{'c':two})=={'c':two}
    with pytest.raises(ValueError): merge_cells({'c':one},{'c':{**one,'value':'conflict'}})

@pytest.mark.parametrize('raw', ['{"a":1}', '```json\n{"a":1}\n```', '<think>private</think>\n{"a":1}'])
def test_json_content_parser(raw):
    assert parse_json_object(raw)=={'a':1}

@pytest.mark.parametrize('raw', ['{"a":', '{"a":1}{"b":2}', 'No JSON'])
def test_json_invalid_does_not_trigger_a_repair_call(raw):
    with pytest.raises(ValueError): parse_json_object(raw)

def test_renderer_handles_empty_answer_and_newlines():
    rendered=render_answer({'explanation':'two\nlines','exact_answer':'','confidence':99})
    assert 'two lines' in rendered and 'Exact Answer: Unable to determine' in rendered
    assert 'Confidence:' not in rendered


def test_json_quotes_containing_think_tags_are_not_rewritten():
    raw='{"quote":"The literal markup <think>example</think> is data."}'
    assert parse_json_object(raw)['quote']=='The literal markup <think>example</think> is data.'
