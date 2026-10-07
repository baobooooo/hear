import json

import pytest

from bcgraph.app import export_result
from bcgraph.dataset import load_queries, selection_ids
from bcgraph.evaluation import import_judge
from bcgraph.metrics import compare, summarize
from bcgraph.storage import Store, atomic_json


def test_ids_file_preserves_cohort_order_and_excludes_answers(tmp_path):
    queries = tmp_path / "questions.jsonl"
    queries.write_text("\n".join(json.dumps({"query_id": str(i), "query": f"Q{i}", "answer": "secret"})
                                 for i in range(3)))
    ids = tmp_path / "selected.ids"
    ids.write_text("2\n0\n")
    assert load_queries(queries, selection_ids(ids_file=str(ids))) == [
        {"query_id": "2", "question": "Q2"}, {"query_id": "0", "question": "Q0"}]
    ids.write_text("0\n0\n")
    with pytest.raises(ValueError):
        selection_ids(ids_file=str(ids))


def test_failed_child_export_preserves_journal_work_and_scope(store):
    store.put("tool:run:q:attempt:1:cell:c:search:1", "a", "success", [{"docid": "11"}])
    store.put("run:q:attempt:1:reader:c:turn:1", "b", "success", {"usage": {
        "reported_prompt_tokens": 100, "completion_tokens": 20, "reused_tokens": 0, "prefilled_tokens": 100}})
    store.put("tool:run:q:attempt:2:cell:c:search:1", "c", "success", [{"docid": "99"}])
    result = export_result({"query_id": "q", "scope": "run:q:attempt:1", "status": "error"},
                           "config", 1000, store=store, model="GLM")
    assert result["retrieved_docids"] == ["11"]
    assert result["tool_call_counts"]["search"] == result["tool_call_counts"]["reader"] == 1
    assert result["usage"]["output_tokens"] == 20
    store.put("run:q:attempt:1:reader:c:turn:2", "d", "ambiguous", {})
    result = export_result({"query_id": "q", "scope": "run:q:attempt:1", "status": "error"}, "config", 1000, store=store)
    assert result["usage"]["output_tokens"] is None
    assert result["metadata"]["uncertain_operations"] == 1


def make_results(path, statuses):
    store = Store(path)
    for ident, status in statuses.items():
        store.write_result({"query_id": ident, "status": status, "result": [{"type": "output_text", "output": "answer"}],
                            "retrieved_docids": ["11"], "metadata": {"duration_ms": 1000}})
    store.event("batch_start")
    store.event("batch_end", wall_seconds=5)
    store.close()


def test_judge_adapter_full_denominator_and_unknowns(tmp_path):
    runs, ev = tmp_path / "run", tmp_path / "eval"
    make_results(runs, {"1": "completed", "2": "unresolved", "3": "completed", "4": "timeout"})
    ev.mkdir()
    nested = ev / "experiment" / "arm"
    atomic_json(nested / "run_1_eval.json", {"query_id": "1", "response": "answer", "is_completed": True,
                                         "judge_result": {"correct": True, "parse_error": False}})
    atomic_json(nested / "run_3_eval.json", {"query_id": "3", "response": "answer", "is_completed": True,
                                         "judge_result": {"parse_error": True}})
    qrels = tmp_path / "qrels.txt"
    qrels.write_text("\n".join(f"{i} 0 11 1" for i in range(1, 5)))
    result = import_judge(str(runs), str(ev), str(tmp_path / "quality"), str(qrels))
    assert result["accuracy"] is None
    assert result["accuracy_lower_bound"] == 0.25
    assert result["retrieval_recall"] == 1
    assert result["categories"] == {"correct": 1, "unresolved": 1, "judge_parse_error": 1, "run_failure": 1}
    labels = str(tmp_path / "quality" / "labels.jsonl")
    assert compare(str(runs), str(runs), labels, labels)["accuracy_a"] is None
    atomic_json(nested / "run_3_eval.json", {"query_id": "3", "response": "answer", "is_completed": True,
                                         "judge_result": {"correct": False, "parse_error": False}})
    result = import_judge(str(runs), str(ev), str(tmp_path / "quality2"), str(qrels))
    assert result["accuracy"] == 0.25
    labels = str(tmp_path / "quality2" / "labels.jsonl")
    paired = compare(str(runs), str(runs), labels, labels)["paired_correctness"]
    assert paired == {"both_correct": 1, "a_only": 0, "b_only": 0, "both_wrong": 3, "mcnemar_exact_p": 1.0}


def test_judge_adapter_rejects_stale_answers_and_missing_manifest_results(tmp_path):
    runs, ev = tmp_path / "run", tmp_path / "eval"
    make_results(runs, {"1": "completed"})
    ev.mkdir()
    atomic_json(ev / "run_1_eval.json", {"query_id": "1", "response": "OLD", "is_completed": True,
                                         "judge_result": {"correct": True}})
    with pytest.raises(ValueError, match="does not match"):
        import_judge(str(runs), str(ev), str(tmp_path / "quality"))
    atomic_json(runs / "_meta" / "manifest.json", {"queries": [{"query_id": "1"}, {"query_id": "2"}]})
    with pytest.raises(ValueError, match="cohort"):
        import_judge(str(runs), str(ev), str(tmp_path / "quality"))


def test_interrupted_batch_is_not_reported_as_complete_wall_time(tmp_path):
    make_results(tmp_path / "run", {"1": "completed"})
    store = Store(tmp_path / "run")
    try:
        assert summarize(store.path)["query_latency_ms_p95"] == 1000
        store.event("batch_start")
        summary = summarize(store.path)
        assert summary["timed_batch_wall_seconds"] is None
        assert summary["throughput_queries_per_minute"] is None
        assert summary["observed_finished_batch_wall_seconds"] == 5
    finally:
        store.close()
