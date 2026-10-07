import json
from copy import deepcopy

import pytest

from bcgraph.graphs import build_graph
from bcgraph.passage_demo import make_demo_runtime, QUESTION
from bcgraph.passages import parse_research_output, report_references, main_reports, registry
from bcgraph.tokenization import Utf8Counter


def test_control_report_and_truncation_preserve_actions():
    control = {'selected_passage_ids': ['P1'], 'next_queries': ['核对出生日期']}
    report = '候选判断：含 "引号" 与 {花括号}。 [P1]\n仍需核查。'
    raw = json.dumps(control) + '\n\nRESEARCH_REPORT\n' + report
    for finish, status in [('stop', 'complete'), ('length', 'truncated')]:
        parsed, notes, partial, actual, state = parse_research_output(raw, finish)
        assert parsed.selected_passage_ids == ['P1']
        assert parsed.next_queries == ['核对出生日期']
        assert actual == report and state == status and not partial


def test_incomplete_control_only_recovers_complete_ids():
    parsed, notes, partial, report, status = parse_research_output(
        '{"selected_passage_ids":["P1","P', 'length')
    assert partial and parsed.selected_passage_ids == ['P1']
    assert not parsed.candidate_answer and not parsed.next_queries
    assert not report and status == 'invalid'
    with pytest.raises(ValueError):
        parse_research_output('{"candidate_answer":"invent', 'length')
    with pytest.raises(ValueError, match='ambiguous'):
        parse_research_output('{}\n{"candidate_answer":"conflict"}', 'stop')


def sources(docid):
    return {docid: {'docid': docid, 'document_sha256': docid + '_hash',
                    'start': 0, 'end': 20, 'text': 'A real original fact'}}


def test_reference_scope_and_main_omission():
    cells = {}
    for cell, docid in [('cell1', 'one'), ('cell2', 'two')]:
        refs, warnings = report_references('Findings [P1] [P999]', sources(docid), 1200)
        assert warnings == ['unknown_report_reference:P999']
        assert set(refs) == {'P1'}
        cells[cell] = {'reader_mode': 'researcher', 'turn': 2, 'report_turn': 2,
                       'latest_research_report': 'Findings [P1] [P999]',
                       'report_status': 'complete', 'report_passage_refs': refs}
    rows = [{**next(iter(registry(sources('two')).values())), 'display_id': 'D1P1'}]
    reports = main_reports(cells, rows, Utf8Counter(), 1200)
    assert '[NOT_SHOWN:cell1/P1]' in reports[0]['unverified_analysis']
    assert '[D1P1]' in reports[1]['unverified_analysis']
    assert '[NOT_SHOWN:cell2/P999]' in reports[1]['unverified_analysis']
    cells['cell2']['report_turn'] = 1
    assert len(main_reports(cells, rows, Utf8Counter(), 1200)) == 1
    assert main_reports(cells, rows, Utf8Counter(), 0) == []


def test_grouped_report_references_use_physical_identity():
    original = {**sources('one'), **sources('two')}
    report = 'Together [P1, P2, P999]; repeat [P1].'
    refs, warnings = report_references(report, original, 1200)
    assert set(refs) == {'P1', 'P2'}
    assert warnings == ['unknown_report_reference:P999']
    cells = {'cell1': {'reader_mode': 'researcher', 'turn': 2, 'report_turn': 2,
             'latest_research_report': report, 'report_status': 'complete',
             'report_passage_refs': refs}}
    shown = next(row for row in registry(original).values()
                 if row['passage_id'] == refs['P2']['passage_id'])
    result = main_reports(cells, [{**shown, 'display_id': 'D3P7'}], Utf8Counter(), 1200)[0]
    assert result['reference_mapping'] == {'P1': None, 'P2': 'D3P7', 'P999': None}
    assert result['unverified_analysis'] == (
        'Together [NOT_SHOWN:cell1/P1], [D3P7], [NOT_SHOWN:cell1/P999]; '
        'repeat [NOT_SHOWN:cell1/P1].')
    assert cells['cell1']['latest_research_report'] == report


@pytest.mark.parametrize('evict', [False, True])
@pytest.mark.parametrize('report_on', [False, True])
async def test_real_graph_raw_chain_reports_and_ablation(store, evict, report_on):
    rt, backend, http = make_demo_runtime(store, evict_once=evict)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.workflow.main_use_research_report = report_on
    raw_responses = []
    def mutate(obj, messages, reader, body):
        if reader:
            raw = json.dumps(obj) + '\nRESEARCH_REPORT\n核查 "身份" {关系} [P1]'
            raw_responses.append(raw)
            return raw
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'research', 'question': QUESTION, 'scope': 'research', 'attempt': 1},
            {'recursion_limit': 256})
        assert state['status'] == 'completed'
        cell = state['cells']['cell1']
        assert cell['latest_research_report'] == '核查 "身份" {关系} [P1]'
        assert cell['report_turn'] == cell['turn'] == 2
        assert [m['content'] for m in cell['raw_history'] if m['role'] == 'assistant'] == raw_responses
        assert cell['report_passage_refs']['P1']['docid'] == 'demo_identity'
        assert cell['metrics'][1]['request_mode'] == ('cold_recovery' if evict else 'chain_delta')
        final = next(m['content'] for m in state['main_history'] if m.get('content', '').startswith('TASK: DECIDE'))
        assert ('research_reports_NOT_EVIDENCE' in final) is report_on
        if report_on:
            assert 'UNVERIFIED ANALYSIS' in final and 'reference_mapping' in final
    finally:
        await rt.close()
        await http.aclose()


@pytest.mark.parametrize('ending', ['missing', 'invalid', 'clear', 'replace', 'truncated'])
async def test_new_round_replaces_report_and_can_revoke_candidate(store, ending):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    turns = []
    def mutate(obj, messages, reader, body):
        if not reader:
            return
        turns.append(obj)
        if len(turns) == 1:
            return json.dumps({**obj, 'candidate_answer': 'Wrong candidate'}) + '\nRESEARCH_REPORT\nOld claim [P1]'
        if ending in {'clear', 'replace'}:
            obj = {**obj, 'clear_candidate': True,
                   'candidate_answer': '' if ending == 'clear' else 'Aurora spectrograph'}
        raw = json.dumps(obj)
        if ending == 'invalid':
            return raw + '\nNOT_A_REPORT'
        if ending == 'truncated':
            return raw + '\nRESEARCH_REPORT\nNew claim [P2]', 'length'
        return raw
    backend.mutate = mutate
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'x', 'question': QUESTION, 'scope': 'replace', 'attempt': 1},
            {'recursion_limit': 256})
        cell = state['cells']['cell1']
        assert state['status'] == 'completed'  # report is never an answer gate
        assert cell['candidate_answer'] == ('' if ending == 'clear' else 'Aurora spectrograph')
        assert cell['report_status'] == (ending if ending in {'invalid', 'truncated'} else 'missing')
        assert cell['latest_research_report'] == ('New claim [P2]' if ending == 'truncated' else '')
        assert any('Old claim' in m.get('content', '') for m in cell['raw_history'])
        assert not any('Old claim' in m.get('content', '') for m in state['main_history'])
        if ending == 'replace':
            assert 'candidate_cleared_then_replaced' in cell['protocol_issues']
    finally:
        await rt.close()
        await http.aclose()


async def test_report_growth_is_not_progress_and_no_known_overflow_fetch(store):
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.reader.max_context_tokens = 40960
    try:
        state = {'query_id': 'x', 'question': QUESTION, 'scope': 'limits', 'attempt': 1}
        state.update(await rt.plan(state))
        cell = rt.init_cell({**state, 'job': state['jobs'][0], 'previous': None})
        cell.update(await rt.cell_fetch(cell))
        cell.update(await rt.cell_read(cell))
        cell['reply']['assistant']['content'] = '{}\nRESEARCH_REPORT\n' + 'No new fact. ' * 100
        cell['no_progress_turns'] = 0
        updates = rt.cell_validate(cell)
        assert updates['no_progress_turns'] == 1
        cell.update(updates)
        cell['raw_history'].append({'role': 'assistant', 'content': 'x' * 40960})
        before = deepcopy(cell['raw_history'])
        async def forbidden(*args, **kwargs):
            raise AssertionError('retrieval must not run after known overflow')
        rt.retrieval.search = forbidden
        fetched = await rt.cell_fetch(cell)
        assert fetched['stop_reason'] == 'context_budget_exhausted'
        assert cell['raw_history'] == before
    finally:
        await rt.close()
        await http.aclose()


async def test_checkpoint_reopen_preserves_report_and_cell_mode(store, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    rt.config.workflow.max_reader_turns = 1
    backend.mutate = lambda obj, messages, reader, body: (
        json.dumps(obj) + '\nRESEARCH_REPORT\nLatest finding [P1]' if reader else None)
    cfg = {'configurable': {'thread_id': 'reports'}, 'recursion_limit': 256}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'cp.sqlite')) as saver:
            graph = build_graph(rt, saver)
            state = await graph.ainvoke(
                {'query_id': 'x', 'question': QUESTION, 'scope': 'checkpoint', 'attempt': 1}, cfg)
            assert state['reopens'] == 1 and state['status'] == 'completed'
            snapshot = await graph.aget_state(cfg)
            assert snapshot.values['cells'] == state['cells']
            assert state['cells']['cell1']['report_turn'] == 2
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'cp.sqlite')) as saver:
            restored = await build_graph(rt, saver).aget_state(cfg)
            assert restored.values['cells'] == state['cells']
        request_count = len([r for r in backend.requests if 'messages' in r])
        replay = await build_graph(rt).ainvoke(
            {'query_id': 'x', 'question': QUESTION, 'scope': 'checkpoint', 'attempt': 1},
            {'recursion_limit': 256})
        assert len([r for r in backend.requests if 'messages' in r]) == request_count
        assert replay['cells']['cell1']['raw_history'] == state['cells']['cell1']['raw_history']
        assert replay['cells']['cell1']['report_passage_refs'] == state['cells']['cell1']['report_passage_refs']
        rt.config.workflow.reader_mode = 'selector'
        with pytest.raises(ValueError, match='lifetime'):
            rt.init_cell({'previous': state['cells']['cell1'], 'job': state['jobs'][0]})
    finally:
        await rt.close()
        await http.aclose()


async def test_gold_canary_not_available_to_runtime(store, tmp_path, monkeypatch):
    from pathlib import Path
    from bcgraph.dataset import load_queries
    canary = 'GOLD_ONLY_CANARY_d8b9b7'
    dataset = tmp_path / 'questions.jsonl'
    dataset.write_text(json.dumps({'query_id': 'x', 'question': QUESTION, 'answer': canary}) + '\n')
    question = load_queries(dataset)[0]
    assert set(question) == {'query_id', 'question'}
    original_read = Path.read_text
    def guarded_read(path, *args, **kwargs):
        if path == dataset:
            raise AssertionError('runtime attempted to read evaluation data')
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded_read)
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    backend.mutate = lambda obj, messages, reader, body: (
        json.dumps(obj) + '\nRESEARCH_REPORT\nSource findings [P1]' if reader else None)
    try:
        state = await build_graph(rt).ainvoke({**question, 'scope': 'canary', 'attempt': 1},
                                             {'recursion_limit': 256})
        assert state['status'] == 'completed'
        assert canary not in json.dumps([state, backend.requests, rt.retrieval.inner.calls])
    finally:
        await rt.close()
        await http.aclose()


async def test_report_history_ambiguous_commit_not_retried(store):
    import httpx
    from bcgraph.config import EndpointConfig
    from bcgraph.transport import ChatClient, AmbiguousCommitError
    calls = []
    history = [{'role': 'system', 'content': 'Research'},
               {'role': 'user', 'content': 'First'},
               {'role': 'assistant', 'content': '{}\nRESEARCH_REPORT\n原文 {fact} [P1]'},
               {'role': 'user', 'content': 'Verify'}]
    def backend(request):
        calls.append(json.loads(request.content))
        raise httpx.ReadTimeout('ambiguous', request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client = ChatClient(EndpointConfig(engine='sparse-vllm', method='h2o', cache='chain'), store, http=http)
        for _ in range(2):
            with pytest.raises(AmbiguousCommitError):
                await client.complete(history, 128, operation_id='report-timeout', writer_key='cell')
        assert len(calls) == 1 and calls[0]['messages'] == history


async def test_main_context_drops_optional_report_before_original_passages(store):
    from bcgraph.passages import final_pool
    from bcgraph.passage_prompts import final_message
    rt, backend, http = make_demo_runtime(store)
    rt.config.workflow.reader_mode = 'researcher'
    backend.mutate = lambda obj, messages, reader, body: (
        json.dumps(obj) + '\nRESEARCH_REPORT\n' + 'Unverified analysis [P1]. ' * 40 if reader else None)
    original_decide = rt.decide
    async def decide_with_tight_context(state):
        cfg = rt.config.workflow
        rows = [{**r, 'display_id': f'D1P{i}'} for i, r in
                enumerate(final_pool(state['cells'], state['question'], cfg.passage_chars), 1)]
        options = rt._research_options(state)
        advertised = [{'cell_id': j['id'], 'next_queries': j['initial_queries'],
                       'read_more': j['read_more']} for _, j in options]
        message = final_message(state['question'], state['cells'], rows, rt._can_reopen(state),
                                rt._research_budget(state), advertised, cfg.answer_policy)
        count = rt.counters['main'].messages([*state['main_history'], message])
        rt.config.main.max_context_tokens = count + 2 * cfg.final_output_tokens + cfg.context_reserve_tokens + 1024 + 8
        return await original_decide(state)
    rt.decide = decide_with_tight_context
    try:
        state = await build_graph(rt).ainvoke(
            {'query_id': 'x', 'question': QUESTION, 'scope': 'tight-main', 'attempt': 1},
            {'recursion_limit': 256})
        assert state['status'] == 'completed'
        message = next(m['content'] for m in state['main_history'] if m.get('content', '').startswith('TASK: DECIDE'))
        assert 'research_reports_NOT_EVIDENCE' not in message
        assert 'Aurora spectrograph' in message
        assert state['decision']['citation_docids'] == ['demo_identity', 'demo_instrument']
    finally:
        await rt.close()
        await http.aclose()


