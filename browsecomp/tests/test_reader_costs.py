from copy import deepcopy
import hashlib
import json

import pytest

from bcgraph.reader_costs import CostTable


def table(dense=10, h2o=5):
    samples = []
    for engine, seconds in [('dense', dense), ('h2o', h2o)]:
        for cache in ('cold', 'warm'):
            for occupancy in (1, 2, 4, 8, 16):
                for query in ('a', 'b', 'c', 'd'):
                    for run in ('r1', 'r2'):
                        samples.append({'sample_id': f'{engine}-{cache}-{occupancy}-{query}-{run}',
                            'query_id': query, 'run': run, 'engine': engine, 'reader_mode': 'selector',
                            'cache_state': cache, 'prompt_tokens': 20000,
                            'new_tokens': 20000 if cache == 'cold' else 4000,
                            'expected_output': 100, 'occupancy': occupancy,
                            'http_seconds': seconds, 'failed': False, 'protocol_error': False})
    return CostTable({'version': 1, 'model': 'GLM-4.7-Flash', 'gain_margin': .1,
        'train_query_ids': ['a', 'b', 'c', 'd'], 'validation_query_ids': ['z'],
        'output_estimates': {'selector:first': {'p75': 100}}, 'samples': samples})


def request(warm=False):
    f = {'reader_mode': 'selector', 'cache_state': 'cold', 'prompt_tokens': 20000,
         'new_tokens': 20000, 'expected_output': 100}
    h = {**f, 'cache_state': 'warm', 'new_tokens': 4000} if warm else dict(f)
    return {'reader': h, 'dense_reader': f, 'h2o_warm': warm, 'h2o_reserve_tokens': 21000}


def loads(dense=0, h2o=0):
    return {'reader': {'active': h2o, 'pending': 0, 'capacity': 8},
            'dense_reader': {'active': dense, 'pending': 0, 'capacity': 8}}


def test_measured_cost_overrides_idle_h2o_load():
    rows = table(h2o=100).select_batch([request()] * 4, loads(dense=4), 200000)
    assert all(backend == 'dense_reader' for backend, _ in rows)


def test_gain_batch_capacity_and_unknown_regions():
    t = table()
    rows = t.select_batch([request()] * 4, loads(), 200000)
    assert all(backend == 'reader' and detail['reason'] == 'measured_gain' for backend, detail in rows)
    assert all(b == 'dense_reader' for b, _ in t.select_batch([request()] * 4, loads(), 0))
    short = request()
    short['reader']['prompt_tokens'] = short['dense_reader']['prompt_tokens'] = 100
    assert t.select_batch([short], loads(), 200000)[0][0] == 'dense_reader'


def test_migration_includes_cold_dense_cost_and_can_leave_h2o():
    assert table().select_batch([request(True)], loads(), 200000)[0][0] == 'reader'
    assert table(h2o=30).select_batch([request(True)], loads(), 200000)[0][0] == 'dense_reader'
    assert table().select_batch([request(True)], loads(), None)[0][0] == 'reader'
    assert table().select_batch([request()], loads(), None)[0][0] == 'dense_reader'


def test_failed_and_protocol_degraded_regions_veto_new_h2o():
    t = table()
    for row in t.data['samples']:
        if row['engine'] == 'h2o':
            row['failed'] = True
    assert t.select_batch([request()], loads(), 200000)[0][0] == 'dense_reader'
    for row in t.data['samples']:
        row['failed'] = False
        row['protocol_error'] = row['engine'] == 'h2o'
    assert t.select_batch([request()], loads(), 200000)[0][0] == 'dense_reader'


def test_hash_leakage_and_no_future_output_dependency(tmp_path):
    t = table()
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(t.data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert CostTable.load(path, digest).output_estimate('selector', 'first', 80) == 80
    with pytest.raises(ValueError, match='hash mismatch'):
        CostTable.load(path, '0' * 64)
    bad = deepcopy(t.data)
    bad['validation_query_ids'].append('a')
    with pytest.raises(ValueError, match='leaks'):
        CostTable(bad)
    bad = deepcopy(t.data)
    bad['samples'][0]['query_id'] = 'z'
    with pytest.raises(ValueError, match='outside training'):
        CostTable(bad)


def test_query_disjoint_neighbours():
    t = table()
    result = t.predict(request()['reader'], 'h2o', 4, exclude_query='a')
    assert result['supported']
    assert all('-a-' not in sample for sample in result['sample_ids'])






@pytest.mark.asyncio
async def test_real_runtime_route_migration_and_journal(tmp_path, store):
    import asyncio
    import httpx
    from bcgraph.config import AppConfig, EndpointConfig, WorkflowConfig
    from bcgraph.runtime import Runtime
    from bcgraph.transport import ChatClient

    class Counter:
        exact = True
        def messages(self, messages):
            return 20000
        def text(self, text):
            return 4000

    calls = []
    def serve(req):
        if req.url.path.endswith('/worker/load'):
            return httpx.Response(200, json={'cache': {'free_slots': 200000}})
        if req.url.path.endswith('/routing_match'):
            return httpx.Response(200, json={'enabled': True, 'present': True, 'state': 'IDLE'})
        payload = json.loads(req.content)
        calls.append((req.url.host, payload))
        return httpx.Response(200, json={'chain_id': 'chain-' + str(len(calls)),
            'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20000, 'completion_tokens': 2}})

    p = tmp_path / 'costs.json'
    p.write_text(json.dumps(table().data))
    dense = EndpointConfig(base_url='http://dense/v1')
    sparse = dense.model_copy(update={'base_url': 'http://h2o/v1', 'engine': 'sparse-vllm',
                                     'method': 'h2o', 'cache': 'chain'})
    cfg = AppConfig(reader=sparse, dense_reader=dense, workflow=WorkflowConfig(
        reader_routing='measured', routing_cost_table=str(p),
        routing_cost_table_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        clients = {k: ChatClient(s, store, http=http) for k, s in [('reader', sparse), ('dense_reader', dense)]}
        rt = Runtime(cfg, store, None, clients, {k: Counter() for k in clients})
        state = {'scope': 'q', 'cell_id': 'cell1', 'backend': 'reader', 'turn': 0, 'handle': None,
            'raw_history': [{'role': 'system', 'content': 'system'}],
            'packed': {'message': {'role': 'user', 'content': 'first'}, 'logical_prompt_tokens': 20000, 'sources': {}},
            'requested_output_tokens': 128, 'output_used': 0, 'sources': {}, 'metrics': []}
        first = await rt.cell_read(state)
        assert first['backend'] == 'reader'
        replay = await rt.cell_read(state)
        assert replay['metrics'][-1]['journal_replay'] and len(calls) == 1
        rt.cost_router.table = table(h2o=30)
        continued = {**state, **first}
        # Follow-up uses the trained first/follow-up prior, never its future actual output.
        rt.cost_router.table.data['output_estimates']['selector:followup'] = {'p75': 100}
        second = await rt.cell_read(continued)
        assert second['backend'] == 'dense_reader'
        assert calls[-1][0] == 'dense' and len(calls[-1][1]['messages']) == 4
        assert 'chain_id' not in calls[-1][1]
        assert second['handle'] is None
