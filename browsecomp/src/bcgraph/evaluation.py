"""Adapt official BrowseComp-Plus Judge artifacts without changing the Judge."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from .metrics import load_results
from .storage import atomic_json


def import_judge(run_dir: str, eval_dir: str, output: str, qrels: str | None = None) -> dict:
    runs = load_results(run_dir)
    ids = [str(row["query_id"]) for row in runs]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected one nonempty run result per unique query ID")
    manifest_path = Path(run_dir) / "_meta" / "manifest.json"
    if manifest_path.exists():
        selected = {str(q["query_id"]) for q in json.loads(manifest_path.read_text())["queries"]}
        if selected != set(ids):
            raise ValueError("Result cohort differs from manifest; missing queries must not disappear from accuracy")
    evaluations = {}
    for path in sorted(Path(eval_dir).rglob("*_eval.json")):
        row = json.loads(path.read_text())
        ident = str(row["query_id"])
        if ident in evaluations:
            raise ValueError(f"Duplicate Judge query ID: {ident}")
        evaluations[ident] = row
    if set(evaluations) - set(ids):
        raise ValueError("Judge directory contains a different query cohort")
    relevant = defaultdict(set)
    if qrels:
        for line in Path(qrels).read_text().splitlines():
            if line.strip():
                ident, _, docid, _ = line.split()
                relevant[ident].add(docid)
    labels, categories, recalls = [], Counter(), []
    for run in runs:
        ident = str(run["query_id"])
        evaluation = evaluations.get(ident)
        correct = None
        if run["status"] != "completed":
            category = "unresolved" if run["status"] == "unresolved" else "run_failure"
            correct = False
        elif evaluation is None:
            category = "judge_missing"
        else:
            # Reject stale Judge results after retry-failed changes an answer.
            response = run["result"][-1]["output"] if run.get("result") else ""
            if evaluation.get("response") != response or not evaluation.get("is_completed"):
                raise ValueError(f"Judge response/status does not match current run: {ident}")
            judge = evaluation.get("judge_result", {})
            if judge.get("parse_error") or type(judge.get("correct")) is not bool:
                category = "judge_parse_error"
            else:
                correct = judge["correct"]
                category = "correct" if correct else "incorrect"
        categories[category] += 1
        labels.append({"query_id": ident, "correct": correct, "category": category})
        if qrels:
            if ident not in relevant:
                raise ValueError(f"Missing qrels for query {ident}")
            recalls.append(len(set(run.get("retrieved_docids", [])) & relevant[ident]) / len(relevant[ident]))
    unknown = sum(row["correct"] is None for row in labels)
    summary = {"query_count": len(runs), "categories": dict(categories),
               "correct_count": categories["correct"], "unknown_judgements": unknown,
               "accuracy": categories["correct"] / len(runs) if not unknown else None,
               "accuracy_lower_bound": categories["correct"] / len(runs),
               "retrieval_recall": statistics.mean(recalls) if recalls else None,
               "judge_directory": str(Path(eval_dir).resolve()),
               "confidence_evaluated": False, "calibration_error": None,
               "note": "Unresolved/run failures remain incorrect in the full denominator; missing/invalid Judge output stays unknown."}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "labels.jsonl").open("x") as file:
        for row in labels:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(destination / "quality.json", summary)
    return summary
