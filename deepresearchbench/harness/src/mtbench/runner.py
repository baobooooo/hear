from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .dataset import load_instance, write_json, write_prediction
from .services import WorkflowServices
from .trajectory import TraceWriter
from .workflow import build_workflow


def make_run_id(instance_id: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"langgraph-t{instance_id}-{stamp}"


async def run_experiment(
    root: Path,
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_id = run_id or make_run_id(config.run.instance_id)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "status.json").exists() or (run_dir / "trajectory.jsonl").exists():
        raise FileExistsError(f"run already contains experiment artifacts: {run_dir}")
    dataset_path = root / "vendor" / "DeepResearchBench" / "data" / "prompt_data" / "query.jsonl"
    instance = load_instance(dataset_path, config.run.instance_id)
    write_json(run_dir / "config.snapshot.json", config.model_dump(mode="json"))
    write_json(run_dir / "instance.json", instance)

    trace = TraceWriter(run_dir / "trajectory.jsonl")
    services = WorkflowServices(config, trace)
    graph = build_workflow(services, report_rounds=config.run.report_rounds)
    started = time.perf_counter()
    await trace.emit(
        "run_started",
        run_id=run_id,
        instance_id=instance["id"],
        subagents=config.run.subagents,
        documents_per_subagent=config.run.documents_per_subagent,
        report_rounds=config.run.report_rounds,
        judge=config.run.judge,
    )
    try:
        result = await graph.ainvoke(
            {
                "instance_id": instance["id"],
                "query": instance["prompt"],
                "language": instance["language"],
                "reports": [],
            },
            config={"recursion_limit": max(32, config.run.report_rounds * 8)},
        )
    except Exception as exc:
        await trace.emit("run_failed", error=type(exc).__name__, detail=str(exc)[:2000])
        write_json(run_dir / "status.json", {"status": "FAILED", "error": str(exc)})
        raise
    finally:
        await services.close()

    elapsed = time.perf_counter() - started
    write_json(run_dir / "plan.json", result["plan"])
    write_json(run_dir / "reports.json", result["reports"])
    write_prediction(run_dir / "prediction.jsonl", instance, result["final_report"])
    status = {
        "status": "COMPLETE",
        "run_id": run_id,
        "instance_id": instance["id"],
        "elapsed_seconds": elapsed,
        "report_count": len(result["reports"]),
        "final_round": result["current_round"],
        "judge_ran": False,
    }
    write_json(run_dir / "status.json", status)
    await trace.emit("run_completed", **status)
    return run_dir, result
