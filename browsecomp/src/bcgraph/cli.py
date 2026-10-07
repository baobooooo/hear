"""CLI for real benchmark runs, protocol demos, endpoint probes and measured comparisons."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid
import httpx

from .app import export_result, open_runtime
from .config import load_config
from .dataset import load_queries, selection_ids
from .demo import QUESTION, make_demo_runtime
from .evidence import stable_hash
from .metrics import compare, summarize
from .storage import Store, atomic_json, redacted_config, source_code_hash
from .transport import completion_url
from .evaluation import import_judge


def run_config(args):
    if getattr(args, 'selection', None):
        from .config_selector import load_selection
        return load_selection(args.selection, allow_synthetic=args.dry_run)
    return load_config(args.config)


@contextmanager
def run_lock(path: Path):
    # GPU execution target is Linux; also works on macOS for demos.
    import fcntl
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def versions() -> dict:
    result = {"python": sys.version}
    for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "langchain-core",
                 "httpx", "mcp", "pydantic", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


async def _gather_queries(coroutines):
    """Do not close shared runtime state while sibling query tasks are alive."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_batch(args):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from .graphs import build_graph
    config = run_config(args)
    queries = load_queries(args.queries, selection_ids(args.ids, args.ids_file), args.limit)
    path = Path(args.output)
    with run_lock(path):
        store = Store(path)
        try:
            fingerprint = stable_hash({"config": config.model_dump(), "queries": queries, "code": source_code_hash()})
            manifest_path = store.meta / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not args.resume:
                    raise ValueError("Output directory already contains a run; use --resume or a new directory")
                if manifest["config_hash"] != fingerprint:
                    raise ValueError("Config/code/cohort changed; refusing to mix incomparable runs")
            else:
                manifest = {"run_id": uuid.uuid4().hex, "config_hash": fingerprint, "code_hash": source_code_hash(),
                            "config": redacted_config(config.model_dump()), "queries": queries,
                            "versions": versions(), "created_at": time.time(), "status": "running"}
                atomic_json(manifest_path, manifest)
            async with open_runtime(config, store) as runtime:
                async with AsyncSqliteSaver.from_conn_string(str(store.meta / "checkpoints.sqlite")) as checkpointer:
                    graph = build_graph(runtime, checkpointer)
                    gate = asyncio.Semaphore(config.workflow.query_concurrency)
                    batch_start = time.monotonic()
                    store.event("batch_start", run_id=manifest["run_id"])
                    async def one(query):
                        result_path = store.result_path(query["query_id"])
                        attempt = 1
                        if result_path.exists():
                            previous = json.loads(result_path.read_text(encoding="utf-8"))
                            if previous["status"] == "completed" or not args.retry_failed:
                                print(f"skip {query['query_id']} ({previous['status']})", flush=True)
                                return
                            attempt = previous["metadata"].get("attempt", 1) + 1
                        scope = f'{manifest["run_id"]}:{stable_hash(query["query_id"])[:16]}:attempt:{attempt}'
                        state = {**query, "scope": scope, "attempt": attempt}
                        graph_config = {"configurable": {"thread_id": scope}, "recursion_limit": 256}
                        async with gate:
                            started = time.monotonic()
                            state["query_deadline"] = time.time() + config.workflow.max_query_seconds
                            store.event("query_start", scope=scope, query_id=query["query_id"])
                            try:
                                snapshot = await graph.aget_state(graph_config)
                                if snapshot.values and not snapshot.next:
                                    output = dict(snapshot.values)
                                else:
                                    deadline = snapshot.values.get("query_deadline", state["query_deadline"])
                                    remaining = deadline - time.time()
                                    if remaining <= 0:
                                        raise TimeoutError("Original query deadline expired before resume")
                                    output = await asyncio.wait_for(
                                        graph.ainvoke(None if snapshot.values else state,
                                                      config=graph_config),
                                        timeout=remaining)
                            except asyncio.CancelledError:
                                store.event("query_cancelled", scope=scope, query_id=query["query_id"])
                                raise
                            except Exception as exc:
                                # Keep the latest committed parent state for diagnosis, but
                                # never represent the exception itself as a solved answer.
                                try:
                                    snapshot = await graph.aget_state(graph_config)
                                    output = {**state, **dict(snapshot.values)}
                                except Exception:
                                    output = dict(state)
                                output.update(status="timeout" if isinstance(
                                    exc, (asyncio.TimeoutError, TimeoutError)) else "error",
                                              errors=[f"{type(exc).__name__}: {exc}"], final_text="", decision={})
                            duration = (time.monotonic() - started) * 1000
                            result = export_result(output, fingerprint, duration, store=store, model=config.main.model)
                            result["metadata"]["timing_scope"] = "current_invocation_active_query"
                            store.write_result(result)
                            atomic_json(store.meta / "trajectories" / (result_path.stem + f".attempt-{attempt}.json"), output)
                            store.event("query_end", scope=scope, query_id=query["query_id"],
                                        status=result["status"], duration_ms=duration)
                            print(f'{query["query_id"]}: {result["status"]}, {duration / 1000:.2f}s', flush=True)
                    try:
                        await _gather_queries(one(query) for query in queries)
                    finally:
                        store.event("batch_end", run_id=manifest["run_id"],
                                    wall_seconds=time.monotonic() - batch_start)
            manifest.update(status="finished", last_finished_at=time.time(), summary=summarize(path))
            atomic_json(manifest_path, manifest)
            atomic_json(store.meta / "summary.json", manifest["summary"])
            print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        finally:
            store.close()


async def run_demo(args):
    from .graphs import build_graph
    path = Path(args.output)
    with run_lock(path):
        if (path / "_meta" / "events.jsonl").exists():
            raise ValueError("Use a new output directory for each demo")
        store = Store(path)
        runtime, backend, http = make_demo_runtime(store, chain=not args.dense,
                                                   evict_once=args.evict_once, two_cells=args.two_cells)
        try:
            start = time.monotonic()
            state = {"query_id": "demo-1", "question": QUESTION, "scope": "synthetic-demo", "attempt": 1}
            output = await build_graph(runtime).ainvoke(state, {"recursion_limit": 256})
            wall = time.monotonic() - start
            store.event("batch_end", wall_seconds=wall)
            store.write_result(export_result(output, "synthetic-not-benchmark", wall * 1000))
            atomic_json(store.meta / "demo_trace.json", output)
            atomic_json(store.meta / "summary.json", summarize(path))
            print(output["final_text"])
            print("Synthetic protocol demo only. No GPU or benchmark accuracy/speed was measured.")
        finally:
            await runtime.close()
            await http.aclose()
            store.close()


async def doctor(args):
    config = load_config(args.config)
    with tempfile.TemporaryDirectory(prefix="bcgraph-doctor-") as directory:
        store = Store(directory)
        try:
            async with open_runtime(config, store) as runtime:
                info = {"versions": versions(), "retrieval": "initialized", "endpoints": {}}
                for name, client in runtime.clients.items():
                    url = completion_url(client.config.base_url).removesuffix("/chat/completions") + "/models"
                    headers = {}
                    import os
                    key = os.environ.get(client.config.api_key_env)
                    if key:
                        headers["Authorization"] = "Bearer " + key
                    response = await client.http.get(url, headers=headers)
                    response.raise_for_status()
                    names = [m["id"] for m in response.json().get("data", [])]
                    if client.config.model not in names:
                        raise ValueError(f"{name}: configured model {client.config.model!r} not in {names}")
                    result = {"model_found": True, "tokenizer": runtime.counters[name].description,
                              "exact_template_count": runtime.counters[name].exact,
                              "configured_engine_method": client.config.method,
                              "method_remotely_verified": False}
                    if args.cache_probe:
                        messages = [{"role": "system", "content": "Answer with only OK."},
                                    {"role": "user", "content": "A protocol test. Reply OK."}]
                        first = await client.complete(messages, 32, operation_id=f"doctor:{name}:1", writer_key=name)
                        second_messages = [*messages, first["assistant"], {"role": "user", "content": "Again, reply OK."}]
                        second = await client.complete(second_messages, 32, operation_id=f"doctor:{name}:2",
                                                       writer_key=name, handle=first["handle"], continuation=True)
                        local_count = runtime.counters[name].messages(messages)
                        reported_count = first["usage"]["reported_prompt_tokens"]
                        if reported_count is not None and reported_count != local_count:
                            raise ValueError(f"{name}: tokenizer/server count mismatch: {local_count} != {reported_count}")
                        result["probe"] = {"request_mode": second["request_mode"], "usage": second["usage"],
                                           "first_prompt_tokens_local": local_count,
                                           "first_prompt_tokens_server": reported_count,
                                           "chain_id_present": bool(second["handle"])}
                        if client.config.cache == "chain" and (not second["handle"] or not (second["usage"]["reused_tokens"] or 0)):
                            raise ValueError("Chain probe did not observe a reusable handle and positive logical reuse")
                    info["endpoints"][name] = result
                print(json.dumps(info, ensure_ascii=False, indent=2))
        finally:
            store.close()


def parser():
    p = argparse.ArgumentParser(description="BrowseComp-Plus persistent-cell LangGraph harness")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run actual LangGraph against configured model and MCP endpoints")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--selection", help="Frozen development-selected configuration; loaded once")
    run.add_argument("--queries", required=True, help="queries.tsv or JSONL with query_id/query; answer fields are ignored")
    run.add_argument("--output", required=True)
    run.add_argument("--ids")
    run.add_argument("--ids-file", help="Existing representative subset, one query ID per line; preserves order")
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-failed", action="store_true", help="With --resume, restart non-completed queries in a NEW attempt/chain")
    run.add_argument("--dry-run", action="store_true")
    demo = sub.add_parser("demo", help="Actual LangGraph with synthetic HTTP/retrieval fixtures, no GPU")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--dense", action="store_true")
    demo.add_argument("--evict-once", action="store_true")
    demo.add_argument("--two-cells", action="store_true")
    doc = sub.add_parser("doctor")
    doc.add_argument("--config", required=True)
    doc.add_argument("--cache-probe", action="store_true", help="Makes two small model requests per endpoint")
    summ = sub.add_parser("summarize")
    summ.add_argument("run_dir")
    comp = sub.add_parser("compare")
    comp.add_argument("run_a")
    comp.add_argument("run_b")
    comp.add_argument("--labels-a", help="JSONL {query_id,correct:boolean}, or JSON id->boolean")
    comp.add_argument("--labels-b")
    judge = sub.add_parser("import-judge", help="Normalize existing official *_eval.json results; no model call")
    judge.add_argument("run_dir")
    judge.add_argument("eval_dir")
    judge.add_argument("--output", required=True, help="New output directory for normalized labels and quality summary")
    judge.add_argument("--qrels", help="Official qrel_evidence.txt, used for retrieval recall including failed queries")
    select = sub.add_parser('select-config', help='Rank development profiles; optionally verify and freeze')
    select.add_argument('--profiles', required=True)
    select.add_argument('--formal-ids', required=True, help='Held-out formal IDs, one per line')
    select.add_argument('--verification', help='Verifier JSON; required to freeze')
    select.add_argument('--output', help='New frozen selection JSON; never overwritten')
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == "run":
            if args.retry_failed and not args.resume:
                raise ValueError("--retry-failed requires --resume")
            if args.dry_run:
                config = run_config(args)
                queries = load_queries(args.queries, selection_ids(args.ids, args.ids_file), args.limit)
                print(json.dumps({"config": redacted_config(config.model_dump()), "queries": queries,
                                  "note": "No endpoint or model called"}, ensure_ascii=False, indent=2))
            else:
                asyncio.run(run_batch(args))
        elif args.command == "demo":
            asyncio.run(run_demo(args))
        elif args.command == "doctor":
            asyncio.run(doctor(args))
        elif args.command == "summarize":
            print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2))
        elif args.command == "compare":
            print(json.dumps(compare(args.run_a, args.run_b, args.labels_a, args.labels_b), ensure_ascii=False, indent=2))
        elif args.command == "import-judge":
            print(json.dumps(import_judge(args.run_dir, args.eval_dir, args.output, args.qrels), ensure_ascii=False, indent=2))
        elif args.command == 'select-config':
            from .config_selector import freeze_selection, rank_profiles
            ids = set(Path(args.formal_ids).read_text().split())
            ranking = rank_profiles(args.profiles, ids)
            if bool(args.output) != bool(args.verification):
                raise ValueError('--output and --verification must be supplied together')
            if args.output:
                verification = json.loads(Path(args.verification).read_text())
                result = freeze_selection(ranking, verification, args.output)
            else:
                result = {**ranking, 'ranked': [{k: v for k, v in r.items() if k != 'config'}
                                               for r in ranking['ranked']]}
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install: pip install -e '.[tokenizer,test]'") from exc
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
