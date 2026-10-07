import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from bcgraph.config import AppConfig, WorkflowConfig, load_config
from bcgraph.graphs import build_graph
from bcgraph.passage_demo import make_demo_runtime, QUESTION
from bcgraph.passage_prompts import final_message
from bcgraph.passages import final_pool, main_reports, report_references
from bcgraph.schemas import FinalDecision
from bcgraph.tokenization import Utf8Counter


def source(name, text='A supported fact'):
    return {name: {'docid': name, 'document_sha256': name, 'title': name,
                   'start': 0, 'end': len(text), 'text': text}}


def cell(name, text):
    sources = source(name)
    refs, _ = report_references(text, sources, 1200)
    return {'focus': name, 'sources': sources, 'reader_mode': 'researcher',
            'turn': 1, 'report_turn': 1, 'report_status': 'complete',
            'latest_research_report': text, 'report_passage_refs': refs}






def test_reports_share_budget_and_reuse_short_report_slack():
    cells = {'cell1': cell('biography', 'Long finding. ' * 500),
             'cell2': cell('award', 'Important contradiction. ' * 500),
             'cell3': cell('work', 'Short.')}
    counter = Utf8Counter()
    reports = main_reports(cells, [], counter, 240, directions=True)
    assert len(reports) == 3
    lengths = [counter.text(r['unverified_analysis']) for r in reports]
    assert sum(lengths) <= 240
    assert lengths[0] > 80 and lengths[1] > 80
    assert [r['objective'] for r in reports] == ['biography', 'award', 'work']
    assert main_reports(cells, [], counter, 0, directions=True) == []
    assert cells['cell1']['latest_research_report'].startswith('Long finding.')
    cells['cell3']['latest_research_report'] = '   '
    assert len(main_reports(cells, [], counter, 240, directions=True)) == 2


def test_passages_interleave_directions_and_keep_shared_provenance():
    a = {**source('a1'), **source('a2'), **source('a3'), **source('shared')}
    b = {**source('b1', 'Counterevidence'), **source('shared')}
    cells = {'cell1': {'sources': a}, 'cell2': {'sources': b}}
    pool = final_pool(cells, 'fact', directions=True)
    assert [r['cell_id'] for r in pool[:2]] == ['cell1', 'cell2']
    shared = [r for r in pool if r['docid'] == 'shared']
    assert len(shared) == 1 and shared[0]['cell_ids'] == ['cell2', 'cell1']
    assert len(pool) == 5


def test_main_sees_failed_direction_and_partial_findings():
    cells = {'cell1': {'focus': 'Biography', 'candidate_answer': '', 'stop_reason': 'ready_for_main'},
             'cell2': {'focus': 'Awards', 'last_error': 'service unavailable', 'stop_reason': 'error'}}
    message = final_message('Question', cells, [], False, {}, [], directions=True)
    data = json.loads(message['content'][message['content'].index('{"question"'):])
    assert data['research_directions'][1]['last_error'] == 'service unavailable'
    assert data['research_directions'][0]['objective'] == 'Biography'
    assert data['candidate_hints_NOT_EVIDENCE'][0]['candidate_answer'] == ''


@pytest.mark.parametrize('count', [2, 3])
async def test_real_graph_combines_partial_directions_with_separate_chains(store, count):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.workflow.research_directions = True
    rt.config.workflow.max_cells_per_query = 3
    captured = []

    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells': [{'focus': f'Direction {i}', 'initial_queries': ['Iris Observatory director 2007']}
                              for i in range(count)]}
        if reader:
            # These local investigations return selected facts, not final answers.
            obj['candidate_answer'] = ''
            return json.dumps(obj) + '\nRESEARCH_REPORT\nLocal supported findings [P1].'
        original = next(m['content'] for m in reversed(messages)
                        if m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES'))
        captured.append(json.loads(original[original.index('{"question"'):]))

    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'directions', 'question': QUESTION, 'scope': 'directions', 'attempt': 1},
            {'recursion_limit': 256})
        assert state['status'] == 'completed'
        assert len(state['cells']) == count
        assert len(backend.chains) == count
        assert all(not c['candidate_answer'] for c in state['cells'].values())
        assert len(captured[-1]['research_directions']) == count
        assert len(captured[-1]['research_reports_NOT_EVIDENCE']) == count
        assert all(c['research_directions'] for c in state['cells'].values())
        assert sum(j['output_limit'] for j in state['jobs']) <= rt.config.workflow.max_total_reader_output_tokens
    finally:
        await http.aclose()


async def test_reopen_selects_second_direction_and_keeps_history(store):
    rt, backend, http = make_demo_runtime(store, two_cells=True)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.workflow.research_directions = True
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'reopen', 'question': QUESTION, 'scope': 'reopen', 'attempt': 1},
            {'recursion_limit': 256})
        before = deepcopy(state['cells']['cell2']['raw_history'])
        decision = FinalDecision(action='research', reopen_cell_id='cell2',
                                 next_queries=['Verify director identity conflict'], explanation='Check identity conflict')
        result = rt._reopen(state, {}, decision)
        assert result['next_job']['id'] == 'cell2'
        restored = rt.init_cell({'job': result['next_job'], 'previous': state['cells']['cell2']})
        assert restored['raw_history'] == before
        assert restored['reopen_instruction'] == 'Check identity conflict'
        decision.reopen_cell_id = 'missing'
        assert rt._reopen(state, {}, decision) is None
        rt.config.workflow.research_directions = False
        with pytest.raises(ValueError, match='cannot change'):
            rt.init_cell({'job': result['next_job'], 'previous': state['cells']['cell2']})
    finally:
        await http.aclose()


@pytest.mark.parametrize('invalid_first_target', [False, True])
async def test_complementary_local_evidence_and_main_targeted_followup(store, invalid_first_target):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.workflow.research_directions = True
    rt.config.workflow.max_cells_per_query = 2
    rt.retrieval.inner.queries['instrument design archive'] = ['demo_instrument']
    rt.retrieval.inner.data['documents']['identity_check'] = {
        'title': 'Designer identity verification',
        'text': 'The Aurora spectrograph designer Eira Stone directed Iris Observatory in 2007.'}
    rt.retrieval.inner.queries['verify instrument identity'] = ['identity_check']
    main_calls = []

    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells': [
                {'focus': 'Identify the director', 'initial_queries': ['Iris Observatory director 2007']},
                {'focus': 'Identify the instrument and its designer', 'initial_queries': ['instrument design archive']}]}
        if reader:
            data = json.loads(messages[-1]['content'])
            obj.update(candidate_answer='', next_queries=[])
            return json.dumps(obj) + '\nRESEARCH_REPORT\n' + data['objective'] + ': findings [P1].'
        original = next(m['content'] for m in reversed(messages)
                        if m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES'))
        data = json.loads(original[original.index('{"question"'):])
        main_calls.append(data)
        if len(main_calls) <= (2 if invalid_first_target else 1):
            return {'action': 'research',
                    'reopen_cell_id': 'missing' if invalid_first_target and len(main_calls) == 1 else 'cell2',
                    'next_queries': ['verify instrument identity'],
                    'explanation': 'Verify that the designer is the director identified by the biography direction.'}

    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'synthesis', 'question': QUESTION, 'scope': 'synthesis', 'attempt': 1},
            {'recursion_limit': 256})
        assert state['status'] == 'completed'
        assert state['reopens'] == 1
        a, b = state['cells']['cell1'], state['cells']['cell2']
        assert {s['docid'] for s in a['sources'].values()} == {'demo_identity'}
        assert {s['docid'] for s in b['sources'].values()} == {'demo_instrument', 'identity_check'}
        assert a['turn'] == 1 and b['turn'] == 2
        assert not a['candidate_answer'] and not b['candidate_answer']
        assert {r['docid'] for r in main_calls[0]['original_passages']} == {'demo_identity', 'demo_instrument'}
        assert 'biography direction' in b['reopen_instruction']
        assert state['decision']['exact_answer'] == 'Aurora spectrograph'
    finally:
        await http.aclose()
