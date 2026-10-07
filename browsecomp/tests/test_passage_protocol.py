from copy import deepcopy
import json
import pytest
from bcgraph.evidence import source_record
from bcgraph.passages import (registry,aliases,Selection,apply_selection,parse_selection,
                             Decision,parse_decision,commit_decision,final_pool)

def sources():
    a=source_record('bio','Eira was born in Cork. Eira studied at Iris College. '*80,0,4000)
    b=source_record('design','Eira designed Aurora.',0,len('Eira designed Aurora.'))
    return {a['source_id']:a,b['source_id']:b}

def test_passages_are_exact_tiling_not_generated_quotes():
    ss=sources(); rows=registry(ss,500)
    for sid,s in ss.items():
        tiles=[r for r in rows.values() if r['source_id']==sid]
        assert ''.join(r['text'] for r in tiles)==s['text']
        for r in tiles: assert r['text']==s['text'][r['start']-s['start']:r['end']-s['start']]

def test_aliases_are_stable_on_append():
    ss=sources(); first=dict(list(ss.items())[:1])
    before=aliases(registry(first)); after=aliases(registry(ss))
    assert all(after[k]==v for k,v in before.items())

def test_invalid_ids_not_fuzzy_matched_and_originals_survive():
    ss=sources(); r=Selection(selected_passage_ids=['P1','P999'])
    chosen,errors,n=apply_selection(r,ss,{},1)
    assert n==1 and errors==['unknown_passage_id:P999']
    assert len(registry(ss))>1
    assert next(iter(chosen.values()))['text']==next(iter(registry(ss).values()))['text']

def test_changed_reader_quote_is_never_used_as_original_text():
    ss=sources()
    reply,notes,partial=parse_selection(json.dumps({'selected_passage_ids':['P1'],
        'quote':'Eira was born in Belfast.','evidence':[{'quote':'FAKE'}]}),'stop')
    chosen,errors,n=apply_selection(reply,ss,{},1)
    assert 'Cork' in next(iter(chosen.values()))['text']
    assert 'Belfast' not in next(iter(chosen.values()))['text']
    assert notes and not partial

@pytest.mark.parametrize('action',['unresolved','research','answer'])
def test_advisory_fields_cannot_kill_valid_decision(action):
    text=json.dumps({'action':action,'exact_answer':'Aurora' if action=='answer' else None,
                     'citations':['P1'],'missing_fact':'Need a date','confidence':0.7,'note':'advisory'})
    d,notes=parse_decision(text)
    assert d.action==action and d.explanation=='Need a date'
    assert notes

def test_one_surplus_brace_allowed_but_two_objects_not():
    d,notes=parse_decision('{"action":"unresolved","explanation":"date missing"}}')
    assert notes
    with pytest.raises(ValueError): parse_decision('{"action":"unresolved"}{"action":"answer"}')

@pytest.mark.parametrize('text',[ '{"selected_passage_ids":["P1",',
                                '{"selected_passage_ids":["P1","P2"'])
def test_partial_ids_are_kept_but_partial_answer_is_not_invented(text):
    r,notes,partial=parse_selection(text,'length')
    assert r.selected_passage_ids and r.candidate_answer=='' and partial
    with pytest.raises(ValueError): parse_selection(text,'stop')

def test_no_nested_salvage():
    with pytest.raises(ValueError): parse_selection('{"broken":{"selected_passage_ids":["P1",','length')

def test_drop_stays_inactive_on_duplicate_selection():
    ss=sources(); chosen,_,_=apply_selection(Selection(selected_passage_ids=['P1']),ss,{},1)
    chosen,_,_=apply_selection(Selection(drop_passage_ids=['P1']),ss,chosen,2)
    chosen,_,changes=apply_selection(Selection(selected_passage_ids=['P1']),ss,chosen,3)
    assert not changes and not next(iter(chosen.values()))['active']

def test_final_citations_are_scope_checked_not_semantic_proof():
    pool=final_pool({'c':{'sources':sources(),'selected_passages':{}}},'instrument')
    d=Decision(action='answer',exact_answer='a normalized nonverbatim answer',citations=['P1'])
    out=commit_decision(d,pool)
    assert out['provenance_checked'] and out['semantic_correctness_verified'] is False
    # This does not claim the answer is correct, unlike the old lexical gate.
    assert out['cited_passages'][0]['docid']==pool[0]['docid']
    for ids in ([],['P999'],['P1','P999']):
        with pytest.raises(ValueError): commit_decision(d.model_copy(update={'citations':ids}),pool)

def test_no_selection_still_provides_real_document_diverse_fallback():
    pool=final_pool({'c':{'sources':sources(),'selected_passages':{}}},'Eira instrument')
    assert {p['docid'] for p in pool[:2]}=={'bio','design'}
    assert all(p['selection']=='source_fallback' for p in pool)
