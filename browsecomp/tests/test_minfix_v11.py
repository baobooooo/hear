"""No benchmark gold answers; tests for the observed contract boundary failures."""
import json
from copy import deepcopy
import pytest
from bcgraph.schemas import parse_plan_output, ReaderReply
from bcgraph.evidence import source_record, accept_reply, validate_final


def plan():
    return {'constraints':[{'id':'c1','description':'Identity clue'},
                           {'id':'target','description':'Requested name','answer_target':True}],
            'cells':[{'id':'cell1','focus':'Investigate','constraint_ids':['c1','target'],
                      'initial_queries':['distinctive clue']}]}


def test_plan_single_redundant_brace_is_unambiguous():
    raw=json.dumps(plan())+'}'
    parsed,notes=parse_plan_output(raw)
    assert [c.id for c in parsed.constraints]==['c1','target']
    assert notes==['discarded_one_redundant_terminal_brace']
    assert raw.endswith('}}')  # Original text is not changed.


def test_plan_trailing_brace_inside_markdown():
    parsed,notes=parse_plan_output('```json\n'+json.dumps(plan())+'}\n```')
    assert notes and len(parsed.cells)==1


@pytest.mark.parametrize('suffix',['{}','}}',' commentary }',']'])
def test_other_tail_is_not_salvaged(suffix):
    with pytest.raises(ValueError): parse_plan_output(json.dumps(plan())+suffix)


def test_single_cell_assignment_is_completed_without_losing_constraints():
    p=plan();p['cells'][0]['constraint_ids']=['target']
    parsed,notes=parse_plan_output(json.dumps(p))
    assert parsed.cells[0].constraint_ids==['target','c1']
    assert len(parsed.constraints)==2 and notes


def test_unknown_assigned_constraint_still_rejected():
    p=plan();p['cells'][0]['constraint_ids']=['typo','target']
    with pytest.raises(ValueError):parse_plan_output(json.dumps(p))


def test_multicell_missing_assignment_still_rejected():
    p=plan();p['cells'][0]['constraint_ids']=['target']
    p['cells'].append({**p['cells'][0],'id':'cell2'})
    with pytest.raises(ValueError):parse_plan_output(json.dumps(p))


def setup():
    constraints={c.id:c.model_dump() for c in parse_plan_output(json.dumps(plan()))[0].constraints}
    text='Eira Example founded an institute in 1988.'
    source=source_record('doc1',text,0,len(text))
    ledger,errors,_=accept_reply(ReaderReply(evidence=[{'candidate':'Eira Example','constraint_id':'c1',
        'source_id':'S1','quote':text}]),{source['source_id']:source},constraints,{},'cell1',1)
    assert not errors
    decision={'candidate':'Eira Example','exact_answer':'Eira Example','evidence_ids':list(ledger)}
    return constraints,ledger,decision


def test_cited_name_does_not_require_redundant_answer_value_field():
    c,e,d=setup()
    assert validate_final(d,e,c)==(True,'answer_with_gaps')
    assert all(v['answer_value'] is None for v in e.values())


def test_strict_coverage_does_not_get_bypassed():
    c,e,d=setup();assert not validate_final(d,e,c,True)[0]


@pytest.mark.parametrize('value',['No Such Person','88','ira','Exampleton'])
def test_not_present_or_partial_word_answer_is_not_accepted(value):
    c,e,d=setup();d['exact_answer']=value
    assert not validate_final(d,e,c)[0]


def test_quote_from_different_candidate_does_not_pass():
    c,e,d=setup();row=next(iter(e.values()));row['candidate_key']='different candidate'
    assert not validate_final(d,e,c)[0]


def test_unsupported_or_retracted_quote_does_not_pass():
    c,e,d=setup();row=next(iter(e.values()));row['active']=False
    assert not validate_final(d,e,c)[0]


def test_unknown_citation_still_rejected():
    c,e,d=setup();d['evidence_ids']=['E999']
    assert not validate_final(d,e,c)[0]


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('answer,expected', [('1948', False), ('1988', True)])
def test_known_target_cannot_be_bypassed_by_other_value_in_quote(strict, answer, expected):
    from bcgraph.schemas import Constraint
    c = {'target': Constraint(id='target', description='Founding year', answer_target=True).model_dump()}
    text = 'Eira was born in 1948 and founded the institute in 1988.'
    source = source_record('doc1', text, 0, len(text))
    e, errors, _ = accept_reply(ReaderReply(evidence=[{
        'candidate': 'Eira', 'constraint_id': 'target', 'source_id': 'S1',
        'quote': text, 'answer_value': '1988'}]),
        {source['source_id']: source}, c, {}, 'cell1', 1)
    assert not errors
    d = {'candidate': 'Eira', 'exact_answer': answer, 'evidence_ids': list(e)}
    assert validate_final(d, e, c, strict)[0] is expected


def test_known_target_requires_citing_target_evidence():
    c, e, d = setup()
    text = 'Eira Example founded the institute.'
    source = source_record('doc2', text, 0, len(text))
    e, errors, _ = accept_reply(ReaderReply(evidence=[{
        'candidate': 'Eira Example', 'constraint_id': 'target',
        'source_id': 'S1', 'quote': source['text'], 'answer_value': 'Eira Example'}]),
        {source['source_id']: source}, c, e, 'cell1', 2)
    assert not errors
    assert not validate_final(d, e, c)[0]
    d['evidence_ids'] = list(e)
    assert validate_final(d, e, c)[0]
