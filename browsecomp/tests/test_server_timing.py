import json

import httpx
import pytest

from bcgraph.config import EndpointConfig
from bcgraph.metrics import distribution, latency_summary
from bcgraph.transport import ChatClient, observed_server_timing


def test_vllm_timing_does_not_mislabel_scheduled_time_or_overall_tps():
    raw = {"time_to_first_token_ms": 200, "queue_time_ms": 100,
           "generation_time_ms": 900, "mean_itl_ms": 100, "tokens_per_second": 9.09}
    timing = observed_server_timing(raw, engine="vllm", output_tokens=10)
    assert timing["server_ttft_ms"] is None
    assert timing["server_prefill_ms"] == 200
    assert timing["server_queue_to_first_token_ms"] == 300
    assert timing["server_tpot_ms"] == 100
    assert timing["server_decode_tokens_per_second"] == 10
    assert timing["server_inference_tokens_per_second"] == 9.09


@pytest.mark.parametrize("count", [None, 0, 1, True])
def test_insufficient_output_never_reports_zero_tpot(count):
    timing = observed_server_timing({"generation_time_ms": 0, "mean_itl_ms": 0},
                                    engine="vllm", output_tokens=count)
    assert timing["server_tpot_ms"] is None
    assert timing["server_decode_tokens_per_second"] is None


@pytest.mark.parametrize("value", [-1, True, "5", float("inf"), float("nan")])
def test_bad_observations_remain_unknown(value):
    timing = observed_server_timing({"generation_time_ms": value}, engine="vllm", output_tokens=10)
    assert timing["server_decode_ms"] is None
    assert timing["server_tpot_ms"] is None
    assert timing["warnings"]


def test_unknown_schema_cannot_be_treated_as_vllm_timing():
    timing = observed_server_timing({"time_to_first_token_ms": 200}, engine="sparse-vllm", output_tokens=10)
    assert timing["server_prefill_ms"] is None
    assert timing["source"] == "unavailable"


def test_sparse_complete_and_partial_token_observations_are_distinct():
    raw = {"schema": "bcgraph.server_timing.v1", "server_ttft_ms": 100,
           "server_decode_ms": 30, "token_timing_complete": True, "token_itl_ms": [10, 20]}
    good = observed_server_timing(raw, engine="sparse-vllm", output_tokens=3)
    assert good["server_tpot_ms"] == 15 and good["token_itl_ms"] == [10, 20]
    partial = observed_server_timing({**raw, "token_itl_ms": [30]}, engine="sparse-vllm", output_tokens=3)
    assert partial["server_ttft_ms"] is None and partial["server_tpot_ms"] is None


def test_itl_is_token_weighted_with_unobserved_requests_reported():
    rows = [
        {"operation_id": "q:main:plan", "success": True, "usage": {"completion_tokens": 4},
         "server_timing": {"token_timing_complete": True, "token_itl_ms": [1, 2, 100]}},
        {"operation_id": "r:main:plan", "success": True, "usage": {"completion_tokens": 3},
         "server_timing": {"token_timing_complete": True, "token_itl_ms": [3, 4]}},
        {"operation_id": "s:main:plan", "success": True, "usage": {"completion_tokens": 5}},
    ]
    itl = latency_summary(rows)["main"]["server_token_itl_ms"]
    assert itl["p50"] == 3 and itl["p95"] == 100
    assert itl["observed_count"] == 5 and itl["expected_intervals_from_known_counts"] == 9
    assert itl["coverage"] == pytest.approx(5 / 9) and itl["missing_or_invalid_count"] == 4
    assert itl["observed_requests"] == 2 and itl["successful_requests"] == 3


def test_percentiles_use_observations_and_report_missing_values():
    result = distribution(list(range(1, 101)) + [None, True, float("nan"), -1])
    assert result["observed_count"] == 100 and result["missing_or_invalid_count"] == 4
    assert result["p50"] == 50 and result["p95"] == 95 and result["p99"] == 99
    assert result["mean"] == 50.5
    assert distribution([])["p95"] is None


def test_success_latency_does_not_hide_failures_or_mix_roles():
    rows = [
        {"operation_id": "q:main:plan", "success": True, "http_ms": 100},
        {"operation_id": "q:main:plan", "success": False, "elapsed_ms": 120000},
        {"operation_id": "q:reader:c:turn:1", "success": True, "http_ms": 200},
    ]
    result = latency_summary(rows)
    assert result["main"]["attempts"] == 2 and result["main"]["failures"] == 1
    assert result["main"]["success_http_ms"]["p95"] == 100
    assert result["main"]["failed_elapsed_ms"]["p95"] == 120000
    assert result["reader"]["success_http_ms"]["p95"] == 200
    assert result["reader"]["server_ttft_ms"]["coverage"] == 0


@pytest.mark.asyncio
async def test_metrics_are_journaled_without_changing_assistant_or_replaying_http(store):
    requests = []
    assistant = {"role": "assistant", "content": " answer ", "reasoning_content": "raw"}
    def backend(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "server-id", "choices": [{"message": assistant, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
            "metrics": {"time_to_first_token_ms": 20, "queue_time_ms": 10, "generation_time_ms": 40},
        })
    config = EndpointConfig(base_url="http://mock/v1", engine="vllm", method="vanilla", cache="prefix")
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client = ChatClient(config, store, http=http)
        kwargs = dict(operation_id="q:main:plan", writer_key="main")
        first = await client.complete([{"role": "user", "content": "question"}], 32, **kwargs)
        replay = await client.complete([{"role": "user", "content": "question"}], 32, **kwargs)
    assert first["assistant"] == assistant and len(requests) == 1
    assert replay["server_timing"] == first["server_timing"] and replay["journal_replay"]
    events = [json.loads(line) for line in (store.meta / "events.jsonl").read_text().splitlines()]
    attempts = [e for e in events if e["kind"] == "model_attempt"]
    assert len(attempts) == 1 and attempts[0]["ttft_ms"] is None
    assert attempts[0]["server_timing"]["server_tpot_ms"] == 20
    assert attempts[0]["server_request_id"] == "server-id"
    assert attempts[0]["client_request_id"] == requests[0].headers["X-Request-ID"]
