import json
from pathlib import Path

import pytest

from bcgraph.config import AppConfig
from bcgraph.graphs import build_graph
from bcgraph.passage_demo import make_demo_runtime, QUESTION
from bcgraph.passage_prompts import direction_output_contract


def r4():
    return json.loads((Path(__file__).parent / 'fixtures/coordinated_config.json').read_text())


def test_r4_budgets_and_report_contract():
    data = r4()
    assert data['reader']['max_context_tokens'] == 202752
    assert data['workflow']['research_sync_interval'] == 2
    assert data['workflow']['max_reopens'] == 0
    assert '250 words' not in direction_output_contract(True)
    assert '250 words' in direction_output_contract(False)
    data['workflow']['max_document_fetches_per_query'] = 240
    with pytest.raises(ValueError, match='document budget'):
        AppConfig.model_validate(data)


def test_operation_phases_do_not_mix_selection_with_reading():
    from bcgraph.metrics import operation_phase
    assert operation_phase('q:reader:c:turn:1:select-documents') == 'document_selection'
    assert operation_phase('q:reader:c:turn:1') == 'reader_initial'
    assert operation_phase('q:reader:c:turn:2') == 'reader_continuation'
    assert operation_phase('q:reader:c:turn:2:delivery-repair') == 'reader_delivery_repair'
    assert operation_phase('q:main:sync:1') == 'main_coordination'


@pytest.mark.parametrize('closing', ['', '\n```'])
def test_real_model_markdown_control_wrapper(closing):
    from bcgraph.passages import parse_research_output
    text = '```json\n{"selected_passage_ids":["P1"]}'+closing+'\nRESEARCH_REPORT\nFinding [P1].'
    selected, notes, partial, report, status = parse_research_output(text, 'stop', bounded=True)
    assert selected.selected_passage_ids == ['P1']
    assert 'markdown_control_fence' in notes
    assert not partial and status == 'complete' and report == 'Finding [P1].'
    with pytest.raises(ValueError):
        parse_research_output('```json\n{"selected_passage_ids":[', 'stop', bounded=True)
    with pytest.raises(ValueError, match='ambiguous'):
        parse_research_output('```json\n{}\n```\n{}\nRESEARCH_REPORT\nFinding.', 'stop', bounded=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('repair_selection', [False, True])
async def test_document_choice_same_chain_and_journal_replay(store, repair_selection):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    selection_calls = []
    selection_payloads = []
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                'initial_queries':['Iris Observatory director 2007']}]}
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            selection_calls.append(body)
            selection_payloads.append(json.loads(messages[-1]['content'].split('\n', 1)[1]))
            if repair_selection and len(selection_calls) == 1:
                return '{"documents":[', 'length'
        if reader and not messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            return json.dumps({'selected_passage_ids':['P1'], 'next_queries':[]})+'\nRESEARCH_REPORT\nAn observation is supported [P1].'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION, 'query_id':'test', 'scope':'test'})
        cell = rt.init_cell({'job':plan['jobs'][0], 'question':QUESTION, 'query_id':'test',
                            'scope':'test', 'constraints':plan['constraints']})
        fetched = await rt.cell_fetch(cell)
        replayed = await rt.cell_fetch(cell)
        assert fetched['raw_history'] == replayed['raw_history']
        assert fetched['output_used'] == replayed['output_used']
        assert fetched['handle']['chain_id']
        cell.update(fetched)
        out = await rt.cell_read(cell)
        assert out['handle']['chain_id'] == cell['handle']['chain_id']
        assert out['turn'] == 1
        assert len(out['metrics']) == 2 + int(repair_selection)
        assert out['metrics'][0]['document_selection']
        if repair_selection:
            assert len(selection_calls) == 2
            assert selection_calls[1]['chain_id'] == cell['handle']['chain_id']
            assert out['metrics'][1]['document_selection_repair']
        for payload in selection_payloads:
            candidates = payload['candidate_documents_NOT_EVIDENCE']
            contract = payload['current_selection_contract']
            assert contract['allowed_document_objects_exact'] == [
                {'docid': item['docid']} for item in candidates]
            assert list(payload)[-1] == 'current_selection_contract'
            assert contract['documents_schema'] == [
                {'docid': 'copy one exact object from allowed_document_objects_exact'}]
        assert out['output_used'] == sum(m['output_budget_charged'] for m in out['metrics'])
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_oversized_catalog_entry_does_not_hide_later_candidates(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    shown = []

    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            payload = json.loads(messages[-1]['content'].split('\n', 1)[1])
            shown.extend(payload['candidate_documents_NOT_EVIDENCE'])
            return {'documents': [{'docid': 'small'}], 'next_queries': []}

    backend.mutate = mutate
    try:
        plan = await rt.plan({'question': QUESTION, 'query_id': 'catalog', 'scope': 'catalog'})
        cell = rt.init_cell({'job': plan['jobs'][0], 'question': QUESTION,
                            'query_id': 'catalog', 'scope': 'catalog', 'constraints': plan['constraints']})
        cell['doc_catalog'] = {'large': {'snippet': 'oversized summary ' * 50000},
                               'small': {'title': 'Useful clue', 'snippet': 'A relevant observation.'}}
        selected, updates = await rt.select_documents(
            cell, [{'docid': 'large'}, {'docid': 'small'}], 2)
        assert [item['docid'] for item in shown] == ['small']
        assert [item['docid'] for item in selected] == ['small']
        assert updates.get('stop_reason') != 'no_new_lead'
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_empty_searches_cannot_create_unbounded_syncs(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify missing instrument','starting_clue_ids':['Q1'],
                'initial_queries':['Iris Observatory missing archive']}]}
    async def empty_search(query):
        return []
    rt.retrieval.inner.search = empty_search
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke({'query_id':'empty','question':QUESTION,'scope':'empty','attempt':1},
                                             {'recursion_limit':64})
        assert state['status'] == 'unresolved'
        assert state['sync_round'] == 2
        assert sum(u['stage'] == 'coordinate' for u in state['main_usage']) == 2
        assert not state.get('next_jobs')
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_unknown_selected_document_is_never_fetched(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                'initial_queries':['Iris Observatory director 2007']}]}
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            return {'documents':[{'docid':'invented'}]}
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke({'query_id':'bad','question':QUESTION,'scope':'bad','attempt':1},
                                             {'recursion_limit':64})
        cell = state['cells']['cell1']
        assert cell['stop_reason'] == 'document_selection_error'
        assert cell['output_used'] > 0
        assert not any(call[0] == 'get_document' for call in rt.retrieval.inner.calls)
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_two_round_sync_resumes_multiple_histories_with_originals(store):
    from bcgraph.retrieval import FixtureRetriever, RecordedRetriever
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    rt.config.workflow.first_documents = rt.config.workflow.followup_documents = 1
    rt.config.workflow.max_reader_turns = 4
    corpus = {'documents': {str(i): {'title':f'Observation {i}',
        'text':f'Iris Observatory evidence {i}: Eira Stone designed the Aurora spectrograph.'} for i in range(8)},
        'search': {f'Iris Observatory director 2007 route {i}': [str(j) for j in range(8)] for i in range(2)}}
    rt.retrieval = RecordedRetriever(FixtureRetriever(corpus), store)
    observed = []
    resumed_jobs = []
    init_cell = rt.init_cell
    def observe_resume(worker):
        if worker.get('previous'):
            resumed_jobs.append(worker['job'])
        return init_cell(worker)
    rt.init_cell = observe_resume
    def mutate(obj, messages, reader, body):
        content = messages[-1]['content']
        if content.startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':f'Independent route {i}', 'starting_clue_ids':['Q1'],
                'initial_queries':[f'Iris Observatory director 2007 route {i}']} for i in range(2)]}
        if content.startswith('TASK: SELECT_DOCUMENTS'):
            data = json.loads(content.split('\n', 1)[1])
            # Choose the last candidate, proving selection is not rank-first fetching.
            return {'documents':[{'docid':data['candidate_documents_NOT_EVIDENCE'][-1]['docid']}]}
        if content.startswith('TASK: COORDINATE_RESEARCH'):
            observed.append(json.loads(content.split('\n', 1)[1]))
        elif reader:
            data = json.loads(content)
            pid = data['source_documents'][0]['passages'][0]['passage_id']
            return json.dumps({'selected_passage_ids':[pid],
                'next_queries':['investigate observation '+str(data['turn'])]}) + '\nRESEARCH_REPORT\nInstrument relationship supported ['+pid+'].'
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke({'query_id':'sync','question':QUESTION,'scope':'sync','attempt':1},
                                             {'recursion_limit':256})
        assert state['status'] == 'completed'
        assert len(observed) == 1
        assert len(observed[0]['eligible_directions']) == 2
        assert len(backend.chains) == 2
        assert resumed_jobs
        for job in resumed_jobs:
            shared = json.loads(job['instruction'])['shared_originals']
            assert shared
            assert [d['docid'] for d in job['shared_documents']] == [d['docid'] for d in shared]
        for c in state['cells'].values():
            assert c['turn'] == 4 and c['revision'] == 2
            assert len(c['metrics']) == 8
            assert 'shared_originals' in c['reopen_instruction']
            for shared in json.loads(c['reopen_instruction'])['shared_originals']:
                assert shared['docid'] in c['doc_catalog']
            assert any(s['docid'] == '7' for s in c['sources'].values())
            assert c['output_used'] == sum(m['output_budget_charged'] for m in c['metrics'])
        assert not any('coordination_error' in e for e in state['errors'])
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_shared_catalog_entry_can_be_selected_without_becoming_local_evidence(store):
    from copy import deepcopy
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    chosen = 'shared-doc'
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            return {'documents': [{'docid': chosen}], 'next_queries': []}

    backend.mutate = mutate
    try:
        job = {'id': 'cell1', 'focus': 'Follow evidence', 'constraint_ids': [],
               'initial_queries': [], 'search_limit': 10, 'document_limit': 10,
               'output_limit': 20000, 'turn_limit': 6}
        worker = {'job': job, 'question': QUESTION, 'query_id': 'shared',
                  'scope': 'shared', 'constraints': {}}
        previous = rt.init_cell(worker)
        previous['doc_catalog']['existing'] = {'docid': 'existing', 'snippet': 'local'}
        snapshot = deepcopy(previous)
        shared = {'docid': 'shared-doc', 'snippet': 'Verified original excerpt',
                  'shared_id': 'S1P1', 'shared_start': 20, 'shared_end': 45}
        resumed = rt.init_cell({**worker, 'previous': previous, 'job': {
            **job, 'instruction': 'Investigate invented-doc',
            'shared_documents': [shared, {'docid': 'existing', 'snippet': 'replacement'}]}})
        assert previous == snapshot
        assert resumed['doc_catalog']['existing']['snippet'] == 'local'
        assert resumed['doc_catalog']['shared-doc'] == shared
        assert 'invented-doc' not in resumed['doc_catalog']
        assert resumed['sources'] == previous['sources'] == {}
        assert resumed['raw_history'] == previous['raw_history']
        selected, updates = await rt.select_documents(resumed, [{'docid': 'shared-doc'}], 1)
        assert selected == [{'docid': 'shared-doc'}]
        assert not updates.get('stop_reason')
        chosen = 'invented-doc'
        rejected = deepcopy(resumed)
        rejected['scope'] = 'shared-reject'
        selected, updates = await rt.select_documents(rejected, [{'docid': 'shared-doc'}], 1)
        assert selected == []
        assert updates['stop_reason'] == 'document_selection_error'
    finally:
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('documents,queries,expected,expected_queries', [
    ([{'docid': 17}, {'docid': {}}, {'docid': 'unknown'}, {'docid': 'hidden'}],
     ['new clue', 3], [{'docid': '17'}], ['new clue']),
    ([{'docid': 'unknown'}], ['new clue'], [], ['new clue']),
    ([], [], [], []),
    ([{'docid': '17'}, {'docid': '17', 'requested_offset': None},
      {'docid': '17', 'requested_offset': 5}, {'docid': '18'}], [],
     [{'docid': '17'}, {'docid': '17', 'requested_offset': 5}], []),
])
async def test_selection_accepts_safe_fields_without_whole_response_repair(
        store, documents, queries, expected, expected_queries):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    calls = []
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            calls.append(messages[-1])
            return json.dumps({'documents': documents, 'next_queries': queries}), 'length'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question': QUESTION, 'query_id': 'partial', 'scope': 'partial'})
        cell = rt.init_cell({'job': plan['jobs'][0], 'question': QUESTION,
                            'query_id': 'partial', 'scope': 'partial', 'constraints': plan['constraints']})
        cell['doc_catalog'] = {i: {'docid': i, 'snippet': 'clue'} for i in ['17', '18', 'hidden']}
        selected, updates = await rt.select_documents(cell, [{'docid': '17'}, {'docid': '18'}], 2)
        assert selected == expected
        assert updates['selection_next_queries'] == expected_queries
        assert not updates.get('stop_reason')
        assert len(calls) == 1
        assert '18' in cell['doc_catalog']
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_selection_request_sequence_advances_without_read_and_replays(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    calls = []
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            calls.append(body)
            return {'documents': [], 'next_queries': []}
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question': QUESTION, 'query_id': 'seq', 'scope': 'seq'})
        cell = rt.init_cell({'job': plan['jobs'][0], 'question': QUESTION,
                            'query_id': 'seq', 'scope': 'seq', 'constraints': plan['constraints']})
        cell['doc_catalog'] = {'17': {'docid': '17', 'snippet': 'clue'}}
        requests = [{'docid': '17'}]
        _, first = await rt.select_documents(cell, requests, 1)
        _, replay = await rt.select_documents(cell, requests, 1)
        assert replay['request_seq'] == first['request_seq'] == 1
        assert replay['raw_history'] == first['raw_history']
        _, second = await rt.select_documents({**cell, **first}, requests, 1)
        assert second['request_seq'] == 2
        assert len(calls) == 2
        assert cell['turn'] == 0
        assert second['handle']['chain_id'] == first['handle']['chain_id']
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_empty_selection_acquires_then_partial_selection_reaches_reader(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    selections = []
    reads = []
    def mutate(obj, messages, reader, body):
        task = messages[-1]['content']
        if task.startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                              'initial_queries':['Iris Observatory director 2007']}]}
        if task.startswith('TASK: SELECT_DOCUMENTS'):
            payload = json.loads(task.split('\n', 1)[1])
            selections.append(payload)
            if len(selections) == 1:
                return {'documents': [], 'next_queries':['another distinctive clue']}
            return {'documents':[{'docid':payload['candidate_documents_NOT_EVIDENCE'][0]['docid']},
                                 {'docid': {'bad': 'id'}}], 'next_queries':[]}
        if reader:
            reads.append(task)
            return '{"selected_passage_ids":["P1"],"next_queries":[]}\nRESEARCH_REPORT\nFound original [P1].'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION,'query_id':'route','scope':'route'})
        cell = rt.init_cell({'job':plan['jobs'][0],'question':QUESTION,'query_id':'route',
                            'scope':'route','constraints':plan['constraints']})
        first = await rt.cell_fetch(cell)
        assert first['fetch_route'] == 'acquire' and first['packed']['message'] is None
        assert not reads and cell['turn'] == 0
        cell.update(first)
        second = await rt.cell_fetch(cell)
        assert second['fetch_route'] == 'read' and second['packed']['sources']
        assert second['searches_used'] > first['searches_used']
        cell.update(second)
        out = await rt.cell_read(cell)
        assert out['turn'] == 1 and out['sources']
        assert len(selections) == 2 and len(reads) == 1
        assert any(source['text'] in reads[0] for source in second['packed']['sources'].values())
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_new_feedback_read_once_even_after_document_budget(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    reads = []
    def mutate(obj, messages, reader, body):
        if reader:
            reads.append(messages[-1]['content'])
            return '{"selected_passage_ids":[],"next_queries":[]}\nRESEARCH_REPORT\nNo evidence for the new check.'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION,'query_id':'feedback','scope':'feedback'})
        cell = rt.init_cell({'job':plan['jobs'][0],'question':QUESTION,'query_id':'feedback',
                            'scope':'feedback','constraints':plan['constraints']})
        cell.update(documents_used=cell['document_limit'], reopen_instruction='Reassess the conflicting claim',
                    revision=2, pending_queries=[])
        packed = await rt.cell_fetch(cell)
        assert packed['fetch_route'] == 'read' and packed['packed']['message']
        assert packed['documents_used'] == cell['document_limit']
        cell.update(packed)
        out = await rt.cell_read(cell)
        assert out['last_consumed_feedback_revision'] == 2
        cell.update(out)
        again = await rt.cell_fetch(cell)
        assert again['packed']['message'] is None
        assert len(reads) == 1
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_graph_routes_empty_selection_to_acquire_before_read(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    selections = 0
    routes = []
    fetch = rt.cell_fetch
    async def observed_fetch(state):
        out = await fetch(state)
        routes.append((state['turn'], out.get('fetch_route'), bool(out.get('packed', {}).get('message'))))
        return out
    rt.cell_fetch = observed_fetch
    def mutate(obj, messages, reader, body):
        nonlocal selections
        if messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                              'initial_queries':['Iris Observatory director 2007']}]}
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            selections += 1
            if selections == 1:
                return {'documents': [], 'next_queries':['a new concrete archive query']}
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke({'query_id':'graph-acquire','question':QUESTION,
                                               'scope':'graph-acquire','attempt':1}, {'recursion_limit':96})
        assert routes[0] == (0, 'acquire', False)
        assert routes[1][0] == 0 and routes[1][1:] == ('read', True)
        assert state['cells'] and state['main_usage']
        assert any(c['sources'] for c in state['cells'].values())
    finally:
        await http.aclose()

@pytest.mark.asyncio
async def test_explicit_historical_read_more_bypasses_current_selection_catalog(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(r4()).workflow
    selections = []
    def mutate(obj, messages, reader, body):
        if messages[-1]['content'].startswith('TASK: SELECT_DOCUMENTS'):
            selections.append(messages[-1]['content'])
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION, 'query_id':'historical-read', 'scope':'historical-read'})
        cell = rt.init_cell({'job':plan['jobs'][0], 'question':QUESTION,
                            'query_id':'historical-read', 'scope':'historical-read',
                            'constraints':plan['constraints']})
        cell.update(pending_queries=[], pending_read_more=[{'docid':'demo_identity','offset':0}],
                    doc_catalog={'demo_identity':{'docid':'demo_identity','snippet':'Historical clue'}})
        fetched = await rt.cell_fetch(cell)
        assert not selections
        assert fetched['fetch_route'] == 'read'
        assert fetched['documents_used'] == 1
        assert any(s['docid'] == 'demo_identity' and s['text'] for s in fetched['packed']['sources'].values())
        cell.update(fetched)
        read = await rt.cell_read(cell)
        assert read['turn'] == 1
        assert any(s['docid'] == 'demo_identity' for s in read['sources'].values())
    finally:
        await http.aclose()
