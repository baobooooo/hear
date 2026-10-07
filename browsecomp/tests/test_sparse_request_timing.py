"""Pure CPU tests; loading this leaf module does not import CUDA dependencies."""
import importlib.util
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "sparseengine-src/sparseengine/entrypoints/openai/request_timing.py"
spec = importlib.util.spec_from_file_location("sparse_request_timing_under_test", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
RequestTiming = module.RequestTiming


def test_nonstream_token_timing_includes_first_and_terminal_tokens():
    timing = RequestTiming(submitted_at=10, queued_at=10.1, admitted_at=10.2)
    timing.observe(1, 11)
    timing.observe(1, 11.2)
    timing.observe(1, 11.5)
    timing.observe(0, 12)  # Already observed final tokens are not counted twice.
    out = timing.metrics(9.5, 3)
    assert out["token_timing_complete"]
    assert out["server_ttft_ms"] == 1500
    assert out["server_decode_ms"] == 500
    assert out["server_tpot_ms"] == 250
    assert out["token_itl_ms"] == pytest.approx([200, 300])
    assert out["dispatcher_prepare_ms"] == pytest.approx(100)
    assert out["dispatcher_admission_wait_ms"] == pytest.approx(100)
    assert out["server_queue_ms"] is None  # Admission is not GPU scheduling.


@pytest.mark.parametrize("counts,total", [([2, 1], 3), ([1, 1], 3), ([], 0)])
def test_coalesced_or_missing_tokens_do_not_fabricate_itl(counts, total):
    timing = RequestTiming(submitted_at=10)
    for index, count in enumerate(counts):
        timing.observe(count, 11 + index)
    out = timing.metrics(10, total)
    assert not out["token_timing_complete"]
    assert out["server_ttft_ms"] is None
    assert out["server_decode_ms"] is None
    assert out["token_itl_ms"] is None


def test_single_token_has_ttft_but_no_tpot():
    timing = RequestTiming(submitted_at=10)
    timing.observe(1, 11)
    out = timing.metrics(10, 1)
    assert out["server_ttft_ms"] == 1000
    assert out["server_tpot_ms"] is None
    assert out["token_itl_ms"] == []


def test_invalid_clock_order_is_marked_incomplete():
    timing = RequestTiming(submitted_at=10)
    timing.observe(1, 11)
    timing.observe(1, 10)
    assert not timing.metrics(10, 2)["token_timing_complete"]


def test_scheduling_is_separate_from_admission_and_first_schedule_is_retained():
    timing = RequestTiming(submitted_at=10, queued_at=10.1, admitted_at=10.2)
    timing.scheduled(True, 10.5)
    timing.scheduled(True, 10.7)
    timing.observe(1, 11)
    timing.scheduled(False, 11.1)
    timing.observe(1, 11.2)
    result = timing.metrics(9.5, 2)
    assert result["dispatcher_admission_wait_ms"] == pytest.approx(100)
    assert result["server_queue_ms"] == pytest.approx(400)
    assert result["server_prefill_ms"] == pytest.approx(500)
    assert result["server_queue_to_first_token_ms"] == pytest.approx(900)
    assert result["prefill_steps"] == 2 and result["decode_steps"] == 1
