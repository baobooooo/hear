from copy import deepcopy
import json

import httpx
import pytest

from bcgraph.config import AppConfig, EndpointConfig, RetrievalConfig, WorkflowConfig, load_config
from bcgraph.app import open_runtime
from bcgraph.runtime import Runtime
from bcgraph.transport import ChatClient


def setup_runtime(store, http, policy="comfort"):
    dense = EndpointConfig(base_url="http://dense/v1", max_inflight=8)
    sparse = dense.model_copy(update={"base_url": "http://sparse/v1", "engine": "sparse-vllm",
        "method": "h2o", "cache": "chain", "max_inflight": 16})
    cfg = AppConfig(reader=sparse, dense_reader=dense,
                    workflow=WorkflowConfig(reader_routing=policy))
    clients = {"reader": ChatClient(sparse, store, http=http),
               "dense_reader": ChatClient(dense, store, http=http)}
    return Runtime(cfg, store, None, clients, {})


def state(tokens=20000):
    return {"backend": "reader", "scope": "q", "cell_id": "c", "turn": 0,
        "raw_history": [{"role": "system", "content": "s"}], "handle": None,
        "packed": {"message": {"role": "user", "content": "u"},
                   "logical_prompt_tokens": tokens, "sources": {}},
        "requested_output_tokens": 128, "output_used": 0, "sources": {}, "metrics": []}


def response():
    return httpx.Response(200, json={"chain_id": "c1", "choices": [{
        "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20000, "completion_tokens": 1}})


@pytest.mark.asyncio
async def test_queue_routing_retains_chain_and_journal_choice(store):
    calls = []
    def handler(request):
        calls.append((request.url.host, json.loads(request.content)))
        return response()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rt = setup_runtime(store, http)
        original = state()
        async with rt.clients["dense_reader"].gate.slot():
            first = await rt.cell_read(original)
        assert first["backend"] == "reader"
        # A changed queue must not change a committed operation's backend.
        again = await rt.cell_read(original)
        assert again["metrics"][-1]["journal_replay"] and len(calls) == 1
        next_state = {**original, **first, "packed": {**original["packed"], "logical_prompt_tokens": 10}}
        assert rt.reader_route(next_state) == ("reader", "retained_chain")
        second = await rt.cell_read(next_state)
        assert second["metrics"][-1]["request_mode"] == "chain_delta"
        assert calls[-1][1]["chain_append_start"] == 1
        stale = deepcopy(next_state)
        stale["handle"]["engine_epoch"] = "old"
        assert rt.reader_route(stale) == ("dense_reader", "short_prompt")


@pytest.mark.asyncio
async def test_ambiguous_request_cannot_move_to_other_backend(store):
    calls = []
    def handler(request):
        calls.append(request.url.host)
        raise httpx.ReadTimeout("lost", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        rt = setup_runtime(store, http)
        async with rt.clients["dense_reader"].gate.slot():
            assert (await rt.cell_read(state()))["last_error"]
        assert (await rt.cell_read(state()))["last_error"]
        assert calls == ["sparse"]


@pytest.mark.asyncio
async def test_short_and_balanced_routes(store):
    async with httpx.AsyncClient() as http:
        rt = setup_runtime(store, http)
        async with rt.clients["dense_reader"].gate.slot():
            assert rt.reader_route(state(100)) == ("dense_reader", "short_prompt")
            assert rt.reader_route(state())[0] == "reader"
        assert rt.reader_route(state())[0] == "dense_reader"
        rt.config.workflow.reader_routing = "balanced"
        async with rt.clients["dense_reader"].gate.slot():
            assert rt.reader_route(state(100))[0] == "reader"
        rt.clients["reader"].config = rt.config.dense_reader
        continued = {**state(), "turn": 1, "backend": "dense_reader"}
        async with rt.clients["dense_reader"].gate.slot():
            assert rt.reader_route(continued) == ("dense_reader", "retained_dense_prefix")


def test_routing_rejects_different_model_and_missing_endpoint():
    with pytest.raises(ValueError, match="requires dense_reader"):
        AppConfig(workflow=WorkflowConfig(reader_routing="balanced"))
    with pytest.raises(ValueError, match="differ in model"):
        AppConfig(dense_reader=EndpointConfig(model="other"),
                  workflow=WorkflowConfig(reader_routing="balanced"))


@pytest.mark.asyncio
async def test_reader_replicas_round_robin_once_and_then_stick(store):
    primary = EndpointConfig(base_url="http://reader-a/v1", engine="sparse-vllm",
                             method="h2o", cache="chain")
    replica = primary.model_copy(update={"base_url": "http://reader-b/v1",
                                         "engine_epoch": "replica-b"})
    config = AppConfig(reader=primary, reader_replicas=[replica])
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        return httpx.Response(200, json={
            "chain_id": request.url.host + "-chain",
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        clients = {
            "reader": ChatClient(primary, store, http=http),
            "reader_replica_1": ChatClient(replica, store, http=http),
        }
        runtime = Runtime(config, store, None, clients, {})
        first_states = []
        for index in range(2):
            initial = state()
            initial.update(scope=f"q{index}", cell_id=f"c{index}", backend="")
            initial["backend"] = runtime.assign_reader_backend(initial)
            first = await runtime.cell_read(initial)
            first_states.append({**initial, **first})
        for first in first_states:
            continued = {**first, "packed": {**first["packed"],
                                              "message": {"role": "user", "content": "next"},
                                              "logical_prompt_tokens": 30}}
            await runtime.cell_read(continued)

    assert [host for host, _ in calls] == [
        "reader-a", "reader-b", "reader-a", "reader-b"]
    assert calls[2][1]["chain_id"] == "reader-a-chain"
    assert calls[3][1]["chain_id"] == "reader-b-chain"
    assert calls[2][1]["chain_append_start"] == 1
    assert calls[3][1]["chain_append_start"] == 1
    assert runtime.assign_reader_backend({"backend": "reader_replica_1"}) == "reader_replica_1"


def test_reader_replicas_reject_mismatch_and_dynamic_routing():
    primary = EndpointConfig(base_url="http://reader-a/v1", engine="sparse-vllm",
                             method="h2o", cache="chain")
    mismatched = primary.model_copy(update={"base_url": "http://reader-b/v1",
                                            "max_context_tokens": 131072})
    with pytest.raises(ValueError, match="differs in max_context_tokens"):
        AppConfig(reader=primary, reader_replicas=[mismatched])
    replica = primary.model_copy(update={"base_url": "http://reader-b/v1"})
    with pytest.raises(ValueError, match="cannot be combined"):
        AppConfig(reader=primary, reader_replicas=[replica],
                  dense_reader=EndpointConfig(base_url="http://dense/v1"),
                  workflow=WorkflowConfig(reader_routing="balanced"))
    with pytest.raises(ValueError, match="differs in max_inflight"):
        AppConfig(reader=primary, reader_replicas=[
            replica.model_copy(update={"max_inflight": primary.max_inflight + 1})])


def test_reader_replicas_load_from_yaml(tmp_path):
    path = tmp_path / "two-readers.yaml"
    path.write_text("""
reader:
  base_url: http://reader-a/v1
  engine: sparse-vllm
  method: h2o
  cache: chain
  engine_epoch: reader-a
  max_inflight: 4
reader_replicas:
  - base_url: http://reader-b/v1
    engine: sparse-vllm
    method: h2o
    cache: chain
    engine_epoch: reader-b
    max_inflight: 4
""")

    config = load_config(path)

    assert config.reader.base_url == "http://reader-a/v1"
    assert [replica.base_url for replica in config.reader_replicas] == [
        "http://reader-b/v1"]


@pytest.mark.asyncio
async def test_open_runtime_gives_reader_replicas_independent_capacity_four(store, tmp_path):
    fixture = tmp_path / "empty.json"
    fixture.write_text(json.dumps({"documents": {}, "search": {}}))
    primary = EndpointConfig(base_url="http://reader-a/v1", engine="sparse-vllm",
                             method="h2o", cache="chain", max_inflight=4)
    replica = primary.model_copy(update={"base_url": "http://reader-b/v1",
                                         "engine_epoch": "reader-b"})
    config = AppConfig(
        reader=primary,
        reader_replicas=[replica],
        retrieval=RetrievalConfig(transport="fixture", fixture_path=str(fixture)),
        workflow=WorkflowConfig(allow_approximate_tokenizer=True),
    )

    async with open_runtime(config, store) as runtime:
        primary_gate = runtime.clients["reader"].gate
        replica_gate = runtime.clients["reader_replica_1"].gate
        assert primary_gate is not replica_gate
        assert primary_gate.snapshot()["capacity"] == 4
        assert replica_gate.snapshot()["capacity"] == 4
