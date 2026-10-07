"""Measured work versus unknown counters; format completion is never called accuracy."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any
from .storage import atomic_json


def percentile95(values: list[float]) -> float | None:
    return sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else None


def distribution(values: list[Any]) -> dict:
    """Nearest-rank percentiles with explicit missing/invalid coverage."""
    known = sorted(float(v) for v in values if not isinstance(v, bool)
                   and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    return {"eligible_count": len(values), "observed_count": len(known),
            "missing_or_invalid_count": len(values) - len(known),
            "coverage": len(known) / len(values) if values else None,
            "mean": statistics.mean(known) if known else None,
            "max": max(known) if known else None,
            **{f"p{p}": known[math.ceil(len(known) * p / 100) - 1] if known else None
               for p in (50, 90, 95, 99)}}


def gpu_energy(rows: list[dict], *, start: float, end: float, max_gap_seconds: float = 10) -> dict:
    """Integrate measured GPU power without filling gaps or extrapolating edges."""
    if end <= start or max_gap_seconds <= 0:
        raise ValueError("Energy window and maximum gap must be positive")
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    samples = sorted((r for r in rows if finite(r.get("timestamp"))), key=lambda r: r["timestamp"])
    joules = covered = 0.0
    for a, b in zip(samples, samples[1:]):
        left, right = max(start, a["timestamp"]), min(end, b["timestamp"])
        dt = b["timestamp"] - a["timestamp"]
        pa, pb = a.get("power_draw_w"), b.get("power_draw_w")
        if (right <= left or dt <= 0 or dt > max_gap_seconds or a.get("error") or b.get("error")
                or not finite(pa) or not finite(pb) or min(pa, pb) < 0):
            continue
        pleft = pa + (pb - pa) * (left - a["timestamp"]) / dt
        pright = pa + (pb - pa) * (right - a["timestamp"]) / dt
        joules += (pleft + pright) * 0.5 * (right - left)
        covered += right - left
    complete = math.isclose(covered, end - start, rel_tol=1e-9, abs_tol=1e-6)
    return {"observed_energy_wh": joules / 3600, "total_energy_wh": joules / 3600 if complete else None,
            "covered_seconds": covered, "window_seconds": end - start,
            "coverage": covered / (end - start), "complete": complete,
            "method": "trapezoid; clipped to window; no extrapolation; GPU power only"}


def latency_summary(attempts: list[dict]) -> dict:
    """Keep roles/phases and failures separate; never add request TPS values."""
    metrics = ("server_ttft_ms", "server_queue_ms", "server_prefill_ms",
               "server_queue_to_first_token_ms", "server_decode_ms", "server_tpot_ms",
               "server_decode_tokens_per_second", "server_inference_tokens_per_second")
    groups = defaultdict(list)
    for row in attempts:
        role = "reader" if ":reader:" in row.get("operation_id", "") else "main"
        for group in (role, role + "/" + operation_phase(row.get("operation_id", ""))):
            groups[group].append(row)
    result = {}
    for name, rows in groups.items():
        success = [r for r in rows if r.get("success")]
        failed = [r for r in rows if not r.get("success")]
        exact_intervals = [r.get("server_timing", {}).get("token_itl_ms") for r in success
                           if r.get("server_timing", {}).get("token_timing_complete") is True]
        exact_intervals = [values for values in exact_intervals if isinstance(values, list)]
        token_counts = [r.get("usage", {}).get("completion_tokens") for r in success]
        known_counts = [n for n in token_counts if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
        itl = distribution([value for values in exact_intervals for value in values])
        expected = sum(max(0, n - 1) for n in known_counts)
        all_counts_known = len(known_counts) == len(success)
        itl.update(observed_requests=len(exact_intervals), successful_requests=len(success),
                   eligible_count=expected if all_counts_known else None,
                   coverage=itl["observed_count"] / expected if all_counts_known and expected else None,
                   missing_or_invalid_count=expected - itl["observed_count"] if all_counts_known else None,
                   expected_intervals_from_known_counts=expected,
                   unknown_output_count_requests=len(success) - len(known_counts),
                   note="Token-weighted observed intervals; absent per-token telemetry is not inferred from mean TPOT.")
        server_distributions = {}
        for metric in metrics:
            eligible = success
            if metric in {"server_tpot_ms", "server_decode_tokens_per_second"}:
                eligible = [r for r in success if r.get("usage", {}).get("completion_tokens") not in (0, 1)]
            server_distributions[metric] = distribution([r.get("server_timing", {}).get(metric) for r in eligible])
            server_distributions[metric]["not_applicable_count"] = len(success) - len(eligible)
        result[name] = {
            "attempts": len(rows), "successes": len(success), "failures": len(failed),
            "success_http_ms": distribution([r.get("http_ms") for r in success]),
            "success_admission_wait_ms": distribution([r.get("admission_wait_ms") for r in success]),
            "failed_elapsed_ms": distribution([r.get("elapsed_ms") for r in failed]),
            "server_token_itl_ms": itl,
            "server_timing_sources": dict(Counter(r.get("server_timing", {}).get("source", "unavailable") for r in success)),
            **server_distributions,
        }
    return result


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(path).glob("run_*.json"))]


def load_events(path: str | Path) -> list[dict]:
    file = Path(path) / "_meta" / "events.jsonl"
    events = []
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash can leave a trailing partial log line. It is not counted.
                continue
    return events


def counter_total(rows: list[dict], name: str) -> dict:
    values = [row.get("usage", {}).get(name) for row in rows]
    known = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return {"total_if_fully_observed": sum(known) if len(known) == len(values) else None,
            "known_sum": sum(known), "unknown_successful_calls": len(values) - len(known)}


def operation_phase(operation_id: str) -> str:
    if ':reader:' in operation_id:
        if ':select-documents' in operation_id:
            return 'document_selection'
        if ':delivery-repair' in operation_id:
            return 'reader_delivery_repair'
        return 'reader_initial' if operation_id.endswith(':turn:1') else 'reader_continuation'
    if ':main:sync:' in operation_id:
        return 'main_coordination'
    return 'main_planning' if ':main:plan' in operation_id else 'main_decision'


def summarize(path: str | Path) -> dict:
    results, events = load_results(path), load_events(path)
    attempts = [e for e in events if e["kind"] == "model_attempt"]
    success = [e for e in attempts if e.get("success")]
    tools = [e for e in events if e["kind"] == "tool_call"]
    starts = sum(e["kind"] == "batch_start" for e in events)
    endings = [e for e in events if e["kind"] == "batch_end"]
    observed_wall = sum(e["wall_seconds"] for e in endings)
    complete_timing = len(endings) > 0 and (starts == len(endings) or starts == 0)
    wall = observed_wall if complete_timing else None
    manifest_path = Path(path) / "_meta" / "manifest.json"
    expected_ids = {str(q["query_id"]) for q in json.loads(manifest_path.read_text())["queries"]} if manifest_path.exists() else None
    actual_ids = [str(r["query_id"]) for r in results]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Duplicate result query IDs")
    cohort_complete = expected_ids == set(actual_ids) if expected_ids is not None else None
    durations = [r.get("metadata", {}).get("duration_ms") for r in results]
    durations = [v for v in durations if v is not None]
    groups = {}
    remedies = [r.get("metadata", {}).get("terminal_remedy", {}) for r in results]
    recoveries = [r for r in results if r.get("metadata", {}).get("terminal_remedy", {}).get("kind") == "output_recovery"
                  and r["metadata"]["terminal_remedy"].get("used")]
    # Parsing a journal replay is not another model failure or another allowance.
    output_failures = {(e["scope"], e["decision_round"]): e for e in events if e["kind"] == "final_output_failure"}
    for role in ("main", "reader"):
        rows = [r for r in success if (":reader:" in r["operation_id"]) == (role == "reader")]
        groups[role] = {"successful_http_calls": len(rows),
                        **{name: counter_total(rows, name) for name in
                           ("reported_prompt_tokens", "completion_tokens", "reused_tokens", "prefilled_tokens")}}
    result = {"query_count": len(results), "status_counts": dict(Counter(r["status"] for r in results)),
              "execution_status_counts": dict(Counter(r.get("metadata", {}).get("execution_status", "unknown") for r in results)),
              "outcome_counts": dict(Counter(r.get("metadata", {}).get("outcome", "unknown") for r in results)),
              "final_format_errors": sum(any(str(e).startswith("final_format_error:") for e in r.get("metadata", {}).get("errors", [])) for r in results),
              "final_output_failure_counts": dict(Counter(e["failure_kind"] for e in output_failures.values())),
              "terminal_remedy_used_queries": sum(bool(r.get("used")) for r in remedies),
              "terminal_remedy_skip_counts": dict(Counter(r["skip_reason"] for r in remedies if r.get("skip_reason"))),
              "output_recovery_attempted_queries": len(recoveries),
              "output_recovery_outcomes": dict(Counter(r["metadata"]["terminal_remedy"].get("outcome", "unknown") for r in recoveries)),
              "reader_length_stops": sum(e.get("finish_reason") == "length" and ":reader:" in e.get("operation_id", "") for e in success),
              "reader_partial_recoveries": sum(bool(e.get("partial_recovery")) for e in events if e["kind"] == "cell_validated"),
              "reader_delivery_repairs": sum(r.get('metadata', {}).get('reader_delivery_repairs', 0) for r in results),
              "source_validation_errors": sum(sum(str(x).startswith(("evidence[", "Cannot retract")) for x in e.get("validation_errors", [])) for e in events if e["kind"] == "cell_validated"),
              "reader_validation_messages": sum(len(e.get("validation_errors", [])) for e in events if e["kind"] == "cell_validated"),
              "confidence_evaluated": False,
              "accuracy": None, "accuracy_note": "Requires official Judge; completed is only supported/formatted output.",
              "model_http_attempts": len(attempts), "successful_http_calls": len(success),
              "failed_http_attempts": len(attempts) - len(success),
              "unknown_work_attempts": sum(bool(e.get("usage_unknown")) for e in attempts),
              "journal_replays": sum(e["kind"] == "model_journal_replay" for e in events),
              "cache_modes": dict(Counter(e.get("request_mode", "unknown") for e in success)),
              "physical_tool_calls": sum(not e.get("cache_hit", False) for e in tools),
              "logical_tool_calls": len(tools),
              "query_latency_ms_mean": statistics.mean(durations) if durations else None,
              "query_latency_ms_median": statistics.median(durations) if durations else None,
               "query_latency_ms_p95": percentile95(durations),
               "query_latency_ms_distribution": distribution([r.get("metadata", {}).get("duration_ms") for r in results]),
               "request_latency_distributions": latency_summary(attempts),
              "timed_batch_wall_seconds": wall,
              "observed_finished_batch_wall_seconds": observed_wall,
              "timing_complete": complete_timing, "batch_invocations": starts or len(endings),
              "cohort_complete": cohort_complete,
              "missing_query_ids": sorted(expected_ids - set(actual_ids)) if expected_ids is not None else None,
              "throughput_queries_per_minute": len(results) * 60 / wall if wall and cohort_complete is not False else None,
              "http_duration_ms_sum": sum(e.get("http_ms", 0) for e in success),
              "admission_wait_ms_sum": sum(e.get("admission_wait_ms", 0) for e in success),
              "cold_recoveries": sum(e.get("request_mode") == "cold_recovery" for e in success),
              "reader_stages": {stage: {
                  "successful_calls": len(rows),
                  "http_ms_p95": percentile95([e["http_ms"] for e in rows if "http_ms" in e]),
                  "http_ms_sum": sum(e.get("http_ms", 0) for e in rows),
                  **{name: counter_total(rows, name) for name in ("reported_prompt_tokens", "completion_tokens", "reused_tokens", "prefilled_tokens")}}
                  for stage in ("initial", "continuation")
                  for rows in [[e for e in success if ":reader:" in e["operation_id"] and
                                (e["operation_id"].endswith(":turn:1")) == (stage == "initial")]]},
              "groups": groups,
              "operation_phases": {phase: {
                  "successful_calls": len(rows),
                  "http_ms_sum": sum(e.get('http_ms', 0) for e in rows),
                  **{name: counter_total(rows, name) for name in
                     ('reported_prompt_tokens', 'completion_tokens', 'reused_tokens', 'prefilled_tokens')}}
                  for phase in sorted({operation_phase(e['operation_id']) for e in success})
                  for rows in [[e for e in success if operation_phase(e['operation_id']) == phase]]},
              "note": "Known token sums exclude unknown work on failed/ambiguous requests. TTFT is not measured by non-streaming HTTP."}
    return result


def _labels(path: str | Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = [{"query_id": k, "correct": v} for k, v in data.items()]
    result, seen = {}, set()
    for row in data:
        ident, value = str(row["query_id"]), row["correct"]
        if ident in seen:
            raise ValueError(f"Duplicate Judge label: {ident}")
        seen.add(ident)
        if value is None:
            continue
        if value is True or value == "yes":
            result[ident] = True
        elif value is False or value == "no":
            result[ident] = False
        else:
            raise ValueError("Judge labels must be boolean or yes/no, not truthy strings")
    return result


def compare(a: str, b: str, labels_a: str | None = None, labels_b: str | None = None) -> dict:
    left = {r["query_id"]: r for r in load_results(a)}
    right = {r["query_id"]: r for r in load_results(b)}
    if not left or set(left) != set(right):
        raise ValueError("Comparison requires exactly the same nonempty query-ID cohort (including failures)")
    sa, sb = summarize(a), summarize(b)
    if sa["cohort_complete"] is False or sb["cohort_complete"] is False:
        raise ValueError("Comparison requires every manifest query, including failures")
    la, lb = _labels(labels_a), _labels(labels_b)
    accuracy_a = sum(la[i] for i in left) / len(left) if set(left) <= la.keys() else None
    accuracy_b = sum(lb[i] for i in right) / len(right) if set(right) <= lb.keys() else None
    wa, wb = sa["timed_batch_wall_seconds"], sb["timed_batch_wall_seconds"]
    paired = None
    if accuracy_a is not None and accuracy_b is not None:
        a_only = sum(la[i] and not lb[i] for i in left)
        b_only = sum(lb[i] and not la[i] for i in left)
        discordant = a_only + b_only
        paired = {"both_correct": sum(la[i] and lb[i] for i in left),
                  "a_only": a_only, "b_only": b_only,
                  "both_wrong": sum(not la[i] and not lb[i] for i in left),
                  "mcnemar_exact_p": min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2 ** discordant)}
    return {"a": sa, "b": sb, "cohort_size": len(left),
            "paired_correctness": paired,
            "wall_speedup_a_over_b": wa / wb if wa and wb else None,
            "accuracy_a": accuracy_a, "accuracy_b": accuracy_b,
            "accuracy_delta_b_minus_a": accuracy_b - accuracy_a if accuracy_a is not None and accuracy_b is not None else None,
            "correct_answers_per_second_a": sum(la[i] for i in left) / wa if accuracy_a is not None and wa else None,
            "correct_answers_per_second_b": sum(lb[i] for i in right) / wb if accuracy_b is not None and wb else None,
            "caution": "Only interpret as a method speedup after matching server/hardware/retriever settings and checking work/quality differences."}
