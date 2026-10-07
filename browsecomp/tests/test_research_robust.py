import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from bcgraph.config import AppConfig, RetrievalConfig
from bcgraph.graphs import build_graph
from bcgraph.passage_demo import QUESTION, make_demo_runtime
from bcgraph.passages import parse_research_output, validate_direction_plan, final_pool, report_references
from bcgraph.retrieval import McpRetriever
from bcgraph.passages import grounded_direction, question_clue_catalog
from bcgraph.passage_prompts import reader_message, DIRECTION_OUTPUT_CONTRACT


def test_direction_output_contract_follows_sources_in_every_turn():
    state = {'question':QUESTION, 'focus':'Identify director', 'search_limit':10,
             'searches_used':0, 'research_direction_guards':True}
    sources = {'s':{'docid':'d','document_sha256':'hash','text':'Original source.','start':0,'end':16}}
    for turn in (0, 2, 3):
        content = reader_message(dict(state, turn=turn), sources)['content']
        payload = json.loads(content)
        assert list(payload)[-1] == 'output_contract'
        assert payload['output_contract'] == DIRECTION_OUTPUT_CONTRACT
        assert payload['source_documents'][0]['passages'][0]['text'] == 'Original source.'
    assert 'output_contract' not in json.loads(reader_message(dict(state, research_direction_guards=False), sources)['content'])


def test_missing_report_marker_preserves_analysis_but_not_ambiguous_control():
    control = '{"selected_passage_ids":["P1"],"candidate_answer":""}'
    text = 'Identity remains uncertain [P1].\nnext_queries=[]'
    parsed, notes, partial, report, status = parse_research_output(control+'\n'+text, 'stop', bounded=True)
    assert report == text and status == 'complete' and not partial
    assert parsed.candidate_answer == '' and parsed.next_queries == []
    assert notes == ['report_marker_missing:preserved_unverified_text']
    assert parse_research_output(control+'\n'+text, 'stop')[4] == 'invalid'
    for second in ('{}', '{"candidate_answer":"different"}'):
        with pytest.raises(ValueError, match='ambiguous'):
            parse_research_output(control+'\n'+second+'\nRESEARCH_REPORT\n'+text, 'stop', bounded=True)


def test_grounded_catalog_queries_and_punctuation():
    question = 'Person 1 published, sometime in 2007. Iris Observatory supplied the instrument.'
    assert question_clue_catalog(question)['Q2'] == 'Iris Observatory supplied the instrument.'
    cell = {'focus':'Identify publication', 'starting_clue_ids':['Q1'],
            'initial_queries':['Person 1 publication']}
    result, notes = grounded_direction(cell, question)
    assert result['starting_clues'] == [question.split('. ')[0] + '.']
    assert result['initial_queries'] == ['published, sometime in 2007.']
    assert 'queries_from_grounded_clues' in notes
    validate_direction_plan([result], question, 3)
    result, notes = grounded_direction(dict(cell, initial_queries=['Person 1', 'Iris Observatory']), question)
    assert result['initial_queries'] == ['Iris Observatory']
    assert notes == ['removed_placeholder_queries']
    with pytest.raises(ValueError, match='supplied Q'):
        grounded_direction(dict(cell, starting_clue_ids=['Q9']), question)
    validate_direction_plan([{'focus':'Publication', 'starting_clues':['published sometime in 2007'],
                              'initial_queries':['publication 2007']}], question, 3)


def profile():
    updates = json.loads((Path(__file__).parent / 'fixtures/workflow_profiles.json').read_text())
    def configured(config, name):
        result = deepcopy(config)
        result['workflow'].update(updates[name])
        return result
    return configured


def test_three_directions_have_full_round_budgets():
    cfg = profile()(AppConfig().model_dump(), 'R3')['workflow']
    assert (cfg['max_document_fetches_per_query'] - cfg['followup_documents']) // 3 >= (
        cfg['first_documents'] + 2 * cfg['followup_documents'])
    assert (cfg['max_total_reader_output_tokens'] - cfg['reader_followup_output_tokens']) // 3 >= (
        cfg['reader_first_output_tokens'] + 2 * cfg['reader_followup_output_tokens'])
    for key, value in [('max_document_fetches_per_query', 30), ('max_total_reader_output_tokens', 7000)]:
        with pytest.raises(ValueError, match='direction .* budget needs'):
            AppConfig.model_validate({'workflow':{**cfg,key:value}})
    with pytest.raises(ValueError, match='direction output budget'):
        AppConfig.model_validate({'workflow':{**cfg,'max_total_reader_output_tokens':30000}})


@pytest.mark.parametrize('repair_good', [True, False])
async def test_last_turn_delivery_repair_preserves_chain_and_is_bounded(store, repair_good):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), 'R3')).workflow
    rt.config.workflow.max_reader_turns = 1
    calls = []
    original = '{"selected_passage_ids":['
    def mutate(obj, messages, reader, body):
        if not reader and messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                              'initial_queries':['Iris Observatory director 2007']}]}
        if reader:
            repair = messages[-1]['content'].startswith('TASK: REPAIR_RESEARCH_DELIVERY')
            calls.append((repair, deepcopy(messages), deepcopy(body)))
            if repair and repair_good:
                return json.dumps({'selected_passage_ids':['P1'], 'next_queries':[]})+'\nRESEARCH_REPORT\nThe director identity is supported [P1]; the instrument still needs verification.'
            return original, 'length'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION,'query_id':'delivery','scope':'delivery'})
        cell = rt.init_cell({'job':plan['jobs'][0], 'question':QUESTION, 'query_id':'delivery',
                             'scope':'delivery','constraints':plan['constraints']})
        cell.update(await rt.cell_fetch(cell))
        before = deepcopy(cell)
        out = await rt.cell_read(cell)
        replayed = await rt.cell_read(before)
        assert replayed['raw_history'] == out['raw_history']
        assert replayed['output_used'] == out['output_used']
        assert replayed['reader_delivery_repairs'] == 1
        cell.update(out)
        validated = rt.cell_validate(cell)
        assert [x[0] for x in calls] == [False, True]
        assert calls[1][1][:-1] == cell['raw_history'][:-2]
        assert cell['raw_history'][-3]['content'] == original
        assert calls[1][2]['chain_id'] and calls[1][2]['chain_append_start'] == 1
        assert out['turn'] == 1 and out['reader_delivery_repairs'] == 1
        assert len(out['metrics']) == 2 and out['metrics'][-1]['delivery_repair']
        assert out['output_used'] == sum(m['output_budget_charged'] for m in out['metrics'])
        assert validated['stop_reason'] == 'turn_budget_exhausted'
        assert validated['report_status'] in {'missing', 'invalid'}
        assert validated['reader_partial_recoveries'] == 0
        assert validated['reader_format_failures'] == int(not repair_good)
    finally:
        await http.aclose()


async def test_delivery_repair_checks_output_and_context_before_call(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), 'R3')).workflow
    state = {'scope':'budget','cell_id':'cell1','turn':1,'backend':'reader','sources':{},
             'reply':{'assistant':{'role':'assistant','content':'{}'},'finish_reason':'stop'},
             'raw_history':[{'role':'assistant','content':'{}'}], 'output_used':100,'output_limit':100,
             'metrics':[], 'handle':None}
    try:
        for output_limit, context_limit in [(100,96000),(10000,128)]:
            state['output_limit'] = output_limit
            rt.clients['reader'].config.max_context_tokens = context_limit
            out = dict(state)
            assert await rt._repair_delivery(state, out) == out
        assert not backend.requests
    finally:
        await http.aclose()


async def test_delivery_check_allows_real_retraction_and_reserves_reopen_budget(store):
    from bcgraph.passages import Selection, apply_selection
    from bcgraph.schemas import FinalDecision
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), 'R3')).workflow
    sources = {'s':{'docid':'d','document_sha256':'hash','text':'Original source.','start':0,'end':16}}
    selected = apply_selection(Selection(selected_passage_ids=['P1']),sources,{},1)[0]
    state = {'sources':sources, 'selected_passages':selected, 'turn':2,
             'reply':{'assistant':{'content':'{"drop_passage_ids":["P1"]}\nRESEARCH_REPORT\nThis passage is not relevant [P1].'},
                      'finish_reason':'stop'}}
    try:
        assert rt._delivery_problem(state) == ''
        cell = {'cell_id':'cell1','focus':'direction','constraint_ids':['target'],'searches_used':3,
                'documents_used':4,'output_used':1000,'turn':3,'seen_queries':[], 'sources':{},'doc_catalog':{}}
        job = rt._reopen_job({'cells':{'cell1':cell},'reopens':0},
                             FinalDecision(action='research',reopen_cell_id='cell1',next_queries=['new query']))
        cfg=rt.config.workflow
        assert job['output_limit']-cell['output_used'] == (cfg.reader_followup_output_tokens+cfg.reader_delivery_repair_tokens)*cfg.reopen_turns
    finally:
        await http.aclose()


@pytest.mark.parametrize('profile_name', ['R3', 'R4'])
async def test_delivery_service_failure_is_charged_and_stops_cell(store, profile_name):
    from bcgraph.transport import ModelRequestError
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), profile_name)).workflow
    state = {'scope':'failure','cell_id':'cell1','turn':1,'backend':'reader','sources':{},
             'reply':{'assistant':{'role':'assistant','content':'not control'},'finish_reason':'stop'},
             'raw_history':[{'role':'assistant','content':'not control'}], 'output_used':100,'output_limit':10000,
             'metrics':[], 'handle':None}
    calls=[]
    async def fail(*args, **kwargs):
        prompt = args[0][-1]['content']
        assert 'Do not write or rewrite RESEARCH_REPORT' in prompt
        assert 'supplied_sources' in prompt
        calls.append(kwargs)
        raise ModelRequestError('synthetic repair failure')
    rt.clients['reader'].complete = fail
    try:
        out = await rt._repair_delivery(state, dict(state))
        assert len(calls) == 1 and calls[0]['continuation']
        assert out['stop_reason'] == 'reader_service_error'
        assert out['last_error'] == 'synthetic repair failure'
        assert out['output_used'] == 100 + rt.config.workflow.reader_delivery_repair_tokens
        assert out['raw_history'] == state['raw_history']
    finally:
        await http.aclose()


def test_report_ranges_map_actual_sources_and_stay_bounded():
    from bcgraph.passages import main_reports, registry, report_aliases
    from bcgraph.tokenization import Utf8Counter
    source = {'s':{'docid':'d','document_sha256':'hash','text':'A'*768,'start':0,'end':768}}
    report = 'Evidence spans [P1-P3], including [P2, P3].'
    refs, warnings = report_references(report, source, 256)
    assert set(refs) == {'P1','P2','P3'} and not warnings
    rows = list(registry(source,256).values())
    rows = [dict(row, display_id=f'D2P{i}') for i,row in enumerate(rows,1)]
    cells = {'cell2':{'reader_mode':'researcher','latest_research_report':report,
                     'report_status':'complete','report_turn':1,'turn':1,'report_passage_refs':refs}}
    mapped = main_reports(cells, rows, Utf8Counter(), 2000)[0]
    assert mapped['reference_mapping'] == {'P1':'D2P1','P2':'D2P2','P3':'D2P3'}
    assert '[P1-P3]' not in mapped['unverified_analysis']
    assert '[D2P1], [D2P2], [D2P3]' in mapped['unverified_analysis']
    assert report_aliases('P1-P999999999') == ['P1-P999999999']
    refs, warnings = report_references('Unbounded [P1-P999999999]',source,256)
    assert not refs and warnings


def test_plan_contract_rejects_dependencies_and_unknown_labels():
    good = {'focus': 'Identify director', 'starting_clues': ['Iris Observatory'],
            'initial_queries': ['Iris Observatory director 2007']}
    validate_direction_plan([good], QUESTION, 3)
    for bad in [dict(good, focus='Find the person identified by the first direction'),
                dict(good, starting_clues=['Invented University']),
                dict(good, initial_queries=['Person 1 biography'])]:
        with pytest.raises(ValueError):
            validate_direction_plan([bad], QUESTION, 3)
    with pytest.raises(ValueError, match='repeats'):
        validate_direction_plan([good, good], QUESTION, 3)


def test_bounded_control_keeps_real_ids_without_inventing_answers():
    raw = json.dumps({'selected_passage_ids': ['P1'] * 100 + [f'P{i}' for i in range(2, 70)] + ['prose', None],
                      'next_queries': ['Check the birth date'], 'candidate_answer': ''})
    with pytest.raises(ValueError):
        parse_research_output(raw, 'stop')
    parsed, notes, partial, _, _ = parse_research_output(raw, 'stop', bounded=True)
    assert parsed.selected_passage_ids == [f'P{i}' for i in range(1, 33)]
    assert not parsed.candidate_answer and parsed.next_queries == ['Check the birth date']
    assert notes and not partial
    with pytest.raises(ValueError, match='ambiguous'):
        parse_research_output('{}\n{}', 'stop', bounded=True)


@pytest.mark.parametrize('top_k', [48, 60, 100])
async def test_real_mcp_larger_search_window(top_k):
    client = await McpRetriever(RetrievalConfig(
        transport='stdio', command=sys.executable,
        args=[str(Path(__file__).parent/'fixtures/mcp_server.py')],
        env={'TEST_MCP_K':'100'}, top_k=top_k)).open()
    try:
        rows = await client.search('test')
        assert len(rows) == top_k
        assert rows[-1]['docid'] == f'd{top_k}'
    finally:
        await client.close()


async def test_real_graph_repairs_plan_and_delivers_labeled_missing_report(store):
    rt, backend, http = make_demo_runtime(store)
    cfg = profile()(rt.config.model_dump(), 'R3')
    # Mutate the existing runtime configuration, retaining its fixture clients.
    rt.config.workflow = AppConfig.model_validate(cfg).workflow
    plans = []
    def mutate(obj, messages, reader, body):
        if not reader and any(m['content'].startswith('TASK: PLAN_RESEARCH') for m in messages) and not any(
                m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES') for m in messages):
            plans.append(messages)
            focus = 'Find author from the first direction' if len(plans) == 1 else 'Identify director and instrument'
            return {'cells':[{'focus':focus, 'starting_clues':['Iris Observatory'],
                              'initial_queries':['Iris Observatory director 2007']}]}
        if reader:
            return obj  # No report: deliver explicitly labeled control summary.
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id':'robust','question':QUESTION,'scope':'robust','attempt':1}, {'recursion_limit':256})
        assert state['status'] == 'completed'
        assert len(plans) == 2
        assert [u['stage'] for u in state['main_usage']][:2] == ['plan','plan_repair']
        assert not any('planner_format_fallback' in e for e in state['errors'])
        cell = state['cells']['cell1']
        assert cell['report_status'] == 'missing' and cell['reader_format_failures'] == 0
        assert cell['latest_research_report'] == ''
        assert all('PROGRAM-GENERATED' not in m['content'] for m in cell['raw_history'] if m['role']=='assistant')
        content = next(m['content'] for m in reversed(state['main_history'])
                       if m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES'))
        data = json.loads(content[content.index('{"question"'):])
        assert not data.get('research_reports_NOT_EVIDENCE')
    finally:
        await http.aclose()


async def test_main_reserves_both_reports_before_filling_original_text(store):
    rt, backend, http = make_demo_runtime(store, two_cells=True)
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id':'pack','question':QUESTION,'scope':'pack','attempt':1}, {'recursion_limit':256})
        rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), 'R3')).workflow
        rt.config.main.max_context_tokens = 20000
        state = deepcopy(state)
        state['scope'] = 'reserve-pack'
        state['main_history'] = state['main_history'][:3]
        state['decision_round'] = 0
        for key, cell in state['cells'].items():
            text = ('Eira Stone designed the Aurora spectrograph. More archive material. ' * 1000)
            cell['sources'][key] = {'docid':key,'document_sha256':key,'title':'Long archive',
                                    'text':text,'start':0,'end':len(text)}
            report = 'Identity findings with original support [P1].\n\n' * 300
            cell.update(reader_mode='researcher', report_status='complete', report_turn=cell['turn'],
                        latest_research_report=report,
                        report_passage_refs=report_references(report,cell['sources'],1200)[0])
        result = await rt.decide(state)
        text = next(m['content'] for m in result['main_history']
                    if m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES'))
        data = json.loads(text[text.index('{"question"'):])
        assert len(data['research_reports_NOT_EVIDENCE']) == 2
        assert all('Identity findings with original support' in r['unverified_analysis']
                   for r in data['research_reports_NOT_EVIDENCE'])
        assert 0 < len(data['original_passages']) < len(final_pool(state['cells'], QUESTION, directions=True))
        request = backend.requests[-1]
        assert rt.counters['main'].messages(request['messages']) + request['max_tokens'] <= 20000
    finally:
        await http.aclose()


@pytest.mark.parametrize('text,finish', [
    ('{"selected_passage_ids":[]}','stop'),
    ('{"selected_passage_ids":["P1","P2","P3","P4","P5"]}\nRESEARCH_REPORT\nPartial analysis', 'length'),
    ('{"selected_passage_ids":["P1","invalid"]}\nRESEARCH_REPORT\nUnverified [P999].','stop'),
])
async def test_valid_control_does_not_rewrite_report(store, text, finish):
    rt, backend, http = make_demo_runtime(store)
    state = {'scope':'no-rewrite','cell_id':'cell1','turn':1,'sources':{},
             'reply':{'assistant':{'role':'assistant','content':text},'finish_reason':finish}}
    try:
        assert rt._delivery_problem(state) == ''
        out = dict(state)
        assert await rt._repair_delivery(state, out) == out
        assert not backend.requests
    finally:
        await http.aclose()


def test_bounded_control_preserves_valid_fields_and_32_passages():
    from bcgraph.passages import parse_research_output
    text = json.dumps({'selected_passage_ids':[f'P{i}' for i in range(1, 34)]+[{}, 'bad'],
                       'candidate_answer':{}, 'next_queries':['good query', 12, ''],
                       'read_more':[{'docid':'known','offset':3}, {'offset':-1}]})
    control, notes, partial, report, status = parse_research_output(text, 'stop', bounded=True)
    assert control.selected_passage_ids == [f'P{i}' for i in range(1, 33)]
    assert control.next_queries == ['good query']
    assert len(control.read_more) == 1
    assert control.candidate_answer == '' and status == 'missing'
    assert not partial and not report and notes

@pytest.mark.parametrize('correction,status', [
    ({'next_queries':['Follow another clue']}, 'truncated'),
    ({'selected_passage_ids':['P1']}, 'stale'),
    ({'candidate_answer':'Different candidate'}, 'stale'),
    ('invalid control again', 'truncated'),
])
async def test_control_repair_preserves_original_report_and_marks_changed_evidence(store, correction, status):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow = AppConfig.model_validate(profile()(rt.config.model_dump(), 'R3')).workflow
    original = 'invalid control\r\nRESEARCH_REPORT\r\nOriginal unfinished analysis [P1].'
    def mutate(obj, messages, reader, body):
        if not reader and messages[-1]['content'].startswith('TASK: PLAN_RESEARCH'):
            return {'cells':[{'focus':'Identify instrument','starting_clue_ids':['Q1'],
                              'initial_queries':['Iris Observatory director 2007']}]}
        if reader:
            if messages[-1]['content'].startswith('TASK: REPAIR_RESEARCH_DELIVERY'):
                supplied = json.loads(messages[-1]['content'].split('\n')[-1])['supplied_sources']
                assert supplied[0]['passage_id'] == 'P1'
                assert supplied[0]['text'] and supplied[0]['docid']
                return correction
            return original, 'length'
    backend.mutate = mutate
    try:
        plan = await rt.plan({'question':QUESTION,'query_id':'preserve','scope':'preserve'})
        cell = rt.init_cell({'job':plan['jobs'][0], 'question':QUESTION, 'query_id':'preserve',
                             'scope':'preserve','constraints':plan['constraints']})
        cell.update(await rt.cell_fetch(cell))
        cell.update(await rt.cell_read(cell))
        result = rt.cell_validate(cell)
        assert result['latest_research_report'] == 'Original unfinished analysis [P1].'
        assert result['report_status'] == status
        assert cell['raw_history'][-3]['content'] == original
        assert cell['reader_delivery_repairs'] == 1
    finally:
        await http.aclose()


def test_bounded_read_more_rejects_bad_types_without_discarding_legal_items():
    raw = json.dumps({'read_more': [
        {'docid':17, 'offset':0}, {'docid':{}, 'offset':1},
        {'docid':'17', 'offset':True}, {'docid':'17', 'offset':-1}],
        'selected_passage_ids':['P1']})
    parsed, notes, *_ = parse_research_output(raw, 'stop', bounded=True)
    assert [r.model_dump() for r in parsed.read_more] == [{'docid':'17', 'offset':0}]
    assert parsed.selected_passage_ids == ['P1']
    assert notes.count('invalid_control_item:read_more') == 3

@pytest.mark.parametrize('control', [
    {'candidate_answer':{}}, {'clear_candidate':'yes'},
    {'selected_passage_ids':{}, 'next_queries':4},
])
def test_all_invalid_supplied_control_is_not_hidden_by_defaults(control):
    with pytest.raises(ValueError, match='no valid control fields'):
        parse_research_output(json.dumps(control), 'stop', bounded=True)

@pytest.mark.parametrize('control', [{}, {'selected_passage_ids':[]},
    {'selected_passage_ids':['P1'], 'candidate_answer':{}}])
def test_valid_empty_or_partial_control_survives_without_invented_fields(control):
    parsed, *_ = parse_research_output(json.dumps(control), 'stop', bounded=True)
    assert parsed.selected_passage_ids == control.get('selected_passage_ids', [])
