import json
import ast
from pathlib import Path
import pytest
from bcgraph.config import AppConfig, WorkflowConfig, EndpointConfig, RetrievalConfig
from bcgraph.app import open_runtime
from bcgraph.tokenization import Utf8Counter
from bcgraph.passage_runtime import PassageRuntime
from bcgraph.runtime import Runtime

@pytest.mark.parametrize('protocol,expected',[('passages-v2',PassageRuntime),('legacy-evidence',Runtime)])
async def test_application_factory_uses_explicit_protocol_not_legacy_by_accident(store,tmp_path,protocol,expected):
    fixture=tmp_path/'empty.json';fixture.write_text(json.dumps({'documents':{},'search':{}}))
    cfg=AppConfig(workflow=WorkflowConfig(research_protocol=protocol,allow_approximate_tokenizer=True),
                  retrieval=RetrievalConfig(transport='fixture',fixture_path=str(fixture)))
    async with open_runtime(cfg,store) as rt:
        assert type(rt) is expected

async def test_counter_reuse_respects_rendering_format(store,tmp_path,monkeypatch):
    import bcgraph.app as app
    made=[]
    def counter(config,*args): made.append(config.tokenizer_format);return Utf8Counter()
    monkeypatch.setattr(app,'make_counter',counter)
    fixture=tmp_path/'empty.json';fixture.write_text(json.dumps({'documents':{},'search':{}}))
    cfg=AppConfig(main=EndpointConfig(tokenizer_path='/same',tokenizer_format='hf_chat_template'),
                  reader=EndpointConfig(tokenizer_path='/same',tokenizer_format='deepseek_v4'),
                  retrieval=RetrievalConfig(transport='fixture',fixture_path=str(fixture)))
    async with open_runtime(cfg,store) as rt:
        assert rt.counters['main'] is not rt.counters['reader']
    assert made==['hf_chat_template','deepseek_v4']

def test_graph_declares_all_new_state_channels_without_importing_fake_langgraph():
    path=Path(__file__).parents[1]/'src/bcgraph/graphs.py'
    tree=ast.parse(path.read_text())
    classes={n.name:{x.target.id for x in n.body if isinstance(x,ast.AnnAssign)}
             for n in tree.body if isinstance(n,ast.ClassDef)}
    assert {'protocol','selected_passages','candidate_answer','missing_fact','selection_turns','passage_chars'}<=classes['CellState']
    assert {'protocol','final_repair_count','final_review_count','delivery_issues','decision_exit_cause'}<=classes['ParentState']


async def test_counter_reuse_separates_engine_null_content_rules(store, tmp_path, monkeypatch):
    import bcgraph.app as app
    made = []

    def counter(config, *args):
        made.append(config.engine)
        return Utf8Counter()

    monkeypatch.setattr(app, 'make_counter', counter)
    fixture = tmp_path / 'empty.json'
    fixture.write_text(json.dumps({'documents': {}, 'search': {}}))
    cfg = AppConfig(
        main=EndpointConfig(engine='vllm', tokenizer_path='/same'),
        reader=EndpointConfig(engine='sparse-vllm', tokenizer_path='/same'),
        retrieval=RetrievalConfig(transport='fixture', fixture_path=str(fixture)),
    )
    async with open_runtime(cfg, store) as runtime:
        assert runtime.counters['main'] is not runtime.counters['reader']
    assert made == ['vllm', 'sparse-vllm']


async def test_runtime_cleanup_does_not_mask_primary_error(store, tmp_path):
    fixture = tmp_path / 'empty.json'
    fixture.write_text(json.dumps({'documents': {}, 'search': {}}))
    cfg = AppConfig(workflow=WorkflowConfig(allow_approximate_tokenizer=True),
                    retrieval=RetrievalConfig(transport='fixture', fixture_path=str(fixture)))
    with pytest.raises(LookupError, match='primary'):
        async with open_runtime(cfg, store) as runtime:
            async def broken_close():
                raise RuntimeError('cleanup')
            runtime.clients['main'].close = broken_close
            raise LookupError('primary')
    assert any(json.loads(line)['kind'] == 'runtime_cleanup_error'
               for line in (store.meta / 'events.jsonl').read_text().splitlines())


async def test_query_gather_settles_siblings_before_raising():
    from bcgraph.cli import _gather_queries
    settled = []
    async def fail():
        raise RuntimeError('query failed')
    async def sibling():
        try:
            await asyncio.Event().wait()
        finally:
            settled.append(True)
    import asyncio
    with pytest.raises(RuntimeError, match='query failed'):
        await _gather_queries([fail(), sibling()])
    assert settled == [True]
