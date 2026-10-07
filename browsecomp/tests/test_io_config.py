import asyncio
import json
from types import SimpleNamespace
import pytest
from bcgraph.config import EndpointConfig, AppConfig, WorkflowConfig, load_config
from bcgraph.dataset import load_queries
from bcgraph.demo import CORPUS
from bcgraph.retrieval import FixtureRetriever, RecordedRetriever, McpRetriever, normalize_docid, parse_search, parse_document, mcp_payload, RetrievalError
from bcgraph.storage import Store, JournalConflict
from bcgraph.tokenization import make_counter, Utf8Counter, prefix_chars
from bcgraph.metrics import counter_total, compare

@pytest.mark.parametrize('values',[{'method':'h2o'}, {'cache':'chain'},
    {'engine':'sparse-vllm','extra_body':{'response_format':{}}},
    {'engine':'sparse-vllm','extra_body':{'seed':0}}])
def test_invalid_engine_combinations(values):
    with pytest.raises(ValueError): EndpointConfig(**values)

def test_real_tokenizer_is_required_by_default():
    with pytest.raises(ValueError): make_counter(EndpointConfig())
    assert make_counter(EndpointConfig(),True).exact is False

def test_environment_config_missing_is_not_silently_blank(tmp_path,monkeypatch):
    monkeypatch.delenv('BC_TEST_MISSING',raising=False)
    path=tmp_path/'c.yaml'; path.write_text('main:\n  tokenizer_path: ${BC_TEST_MISSING}\n')
    with pytest.raises(ValueError): load_config(path)
    monkeypatch.setenv('BC_TEST_MISSING','/local/tokenizer')
    assert load_config(path).main.tokenizer_path=='/local/tokenizer'

def test_dataset_does_not_expose_gold_answer(tmp_path):
    path=tmp_path/'q.jsonl'; path.write_text(json.dumps({'query_id':'q1','query':'Question?','answer':'SECRET'})+'\n')
    assert load_queries(path)==[{'query_id':'q1','question':'Question?'}]

def test_duplicate_query_ids_fail(tmp_path):
    path=tmp_path/'q.tsv'; path.write_text('x\tA\nx\tB\n')
    with pytest.raises(ValueError): load_queries(path)

@pytest.mark.parametrize('value',[[{'docid':3,'snippet':'s'}],{'result':[{'docid':3,'snippet':'s'}]}, {'results':[{'docid':3,'snippet':'s'}]}])
def test_search_compatibility(value): assert parse_search(value)[0]['docid']=='3'

@pytest.mark.parametrize('value', [None, True, False, 17.0, [], {}, '', '  '])
def test_invalid_document_id_types(value):
    with pytest.raises(RetrievalError): normalize_docid(value)
    with pytest.raises(RetrievalError): parse_search([{'docid': value, 'snippet': 's'}])
    with pytest.raises(RetrievalError): parse_document({'docid': value, 'text': 's'})

def test_document_ids_preserve_opaque_strings():
    assert normalize_docid(17) == normalize_docid('17')
    assert normalize_docid('0017') != normalize_docid(17)
    assert normalize_docid(' P17 ') == ' P17 '

@pytest.mark.asyncio
async def test_document_return_id_is_normalized_and_checked():
    from bcgraph.config import RetrievalConfig
    retriever = McpRetriever(RetrievalConfig())
    async def same(name, arguments):
        assert arguments == {'docid': '17'}
        return {'docid': 17, 'text': 'original'}
    retriever._call = same
    assert (await retriever.get_document('17'))['docid'] == '17'
    async def different(name, arguments):
        return {'docid': '0017', 'text': 'another document'}
    retriever._call = different
    with pytest.raises(RetrievalError, match='different docid'):
        await retriever.get_document(17)

@pytest.mark.asyncio
async def test_mcp_timeout_is_local_and_session_remains_usable(fake_mcp):
    from bcgraph.config import RetrievalConfig
    client = await McpRetriever(RetrievalConfig(timeout_seconds=.03)).open()
    try:
        with pytest.raises(RetrievalError, match='timed out after 0.03 seconds'):
            await client.search('slow')
        assert not asyncio.current_task().cancelling()
        assert fake_mcp.cancelled.is_set()
        assert await client.search('fast') == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_caller_cancellation_does_not_cancel_shared_session(fake_mcp):
    from bcgraph.config import RetrievalConfig
    client = await McpRetriever(RetrievalConfig(timeout_seconds=1, max_inflight=2)).open()
    try:
        call = asyncio.create_task(client.search('slow'))
        await fake_mcp.entered.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert fake_mcp.cancelled.is_set()
        assert not client.owner_task.done()
        assert await client.search('fast') == []
        assert not client.requests
    finally:
        await client.close()


def test_document_and_mcp_payload():
    assert parse_document({'result':{'docid':1,'text':'original'}})['docid']=='1'
    assert parse_document(None) is None
    response=SimpleNamespace(isError=False,structuredContent=None,content=[SimpleNamespace(type='text',text='{"result":null}')])
    assert mcp_payload(response)=={'result':None}
    with pytest.raises(RetrievalError): parse_search({'unexpected':[]})

@pytest.mark.asyncio
async def test_tool_memory_cache_and_persistent_replay(store):
    inner=FixtureRetriever(CORPUS); recorded=RecordedRetriever(inner,store,1)
    a,b=await asyncio.gather(recorded.search('Iris Observatory director 2007','op1'),
                             recorded.search('Iris Observatory director 2007','op2'))
    assert a==b and len(inner.calls)==1
    b[0]['docid']='mutated'
    assert (await recorded.search('Iris Observatory director 2007','op1'))[0]['docid']=='demo_identity'
    assert len(inner.calls)==1

def test_journal_input_conflict_and_path_safety(store):
    store.put('a','f','success',{'ok':1})
    with pytest.raises(JournalConflict): store.get('a','other')
    with pytest.raises(JournalConflict): store.put('a','f','ambiguous')
    assert store.result_path('../../escape').parent==store.path

def test_unicode_source_prefix_preserved():
    text='中文 source 🌟'*100
    n=prefix_chars(text,97,Utf8Counter())
    assert n>0 and len(text[:n].encode())<=97

def test_unknown_aggregated_counter_is_not_zero():
    aggregate=counter_total([{'usage':{'reused_tokens':10}},{'usage':{}}],'reused_tokens')
    assert aggregate=={'total_if_fully_observed':None,'known_sum':10,'unknown_successful_calls':1}

def test_compare_refuses_different_cohorts(tmp_path):
    a=Store(tmp_path/'a'); b=Store(tmp_path/'b')
    try:
        a.write_result({'query_id':'a','status':'completed'}); b.write_result({'query_id':'b','status':'completed'})
        with pytest.raises(ValueError): compare(str(a.path),str(b.path))
    finally: a.close(); b.close()


def test_tsv_rejects_extra_gold_label_column(tmp_path):
    path=tmp_path/'q.tsv'; path.write_text('1\tQuestion?\tSECRET ANSWER\n')
    with pytest.raises(ValueError): load_queries(path)

@pytest.mark.asyncio
async def test_recorded_retriever_checks_document_identity_before_journaling(store):
    inner = FixtureRetriever(CORPUS)
    async def fetch(docid):
        return {'docid': 17 if docid == '17' else 'different', 'text': 'Original.'}
    inner.get_document = fetch
    recorded = RecordedRetriever(inner, store)
    assert (await recorded.get_document(17, 'normalized'))['docid'] == '17'
    for _ in range(2):
        with pytest.raises(RetrievalError, match='different docid'):
            await recorded.get_document('0017', 'mismatch')
