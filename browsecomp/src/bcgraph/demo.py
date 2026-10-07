"""Deterministic in-process HTTP fixture. It simulates protocols, NOT GPU speed/accuracy."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import httpx
from .admission import PriorityGate
from .config import AppConfig, EndpointConfig, RetrievalConfig, WorkflowConfig
from .retrieval import FixtureRetriever, RecordedRetriever
from .runtime import Runtime
from .storage import Store
from .tokenization import Utf8Counter
from .transport import ChatClient

QUESTION = "Which instrument did the person who directed Iris Observatory in 2007 design?"
CORPUS = {"documents": {
    "demo_identity": {"title": "Iris Observatory annual report", "text": "In 2007, Eira Stone was the director of Iris Observatory. The annual report describes its telescope maintenance program."},
    "demo_instrument": {"title": "Instrument design archive", "text": "Eira Stone designed the Aurora spectrograph. The instrument measures the spectra of nearby stars."}},
    "search": {"Iris Observatory director 2007": ["demo_identity"],
               "Eira Stone designed instrument": ["demo_instrument"]}}


class DemoHTTP:
    def __init__(self, chain: bool = True, evict_once: bool = False, two_cells: bool = False):
        self.chain, self.evict_once, self.two_cells = chain, evict_once, two_cells
        self.evicted = False
        self.chains: dict[str, list[dict]] = {}
        self.requests: list[dict] = []
        self.sequence = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "demo-model"}]})
        body = json.loads(request.content)
        self.requests.append(deepcopy(body))
        messages = body["messages"]
        is_reader = request.url.host == "mock-reader"
        chain_id, reused = body.get("chain_id"), 0
        if chain_id:
            if self.evict_once and not self.evicted:
                self.evicted = True
                self.chains.pop(chain_id, None)
                return httpx.Response(410, json={"detail": {"code": "chain_gone", "message": "Test eviction"}})
            previous = self.chains.get(chain_id)
            if previous is None:
                return httpx.Response(404, json={"detail": {"code": "chain_not_found", "message": "Not present"}})
            if body.get("chain_append_start") == 1:
                if messages[0] != previous[-1]:
                    return httpx.Response(409, json={"detail": {"code": "chain_prefix_mismatch", "message": "Raw assistant was changed"}})
                logical = [*previous, *messages[1:]]
            else:
                if messages[:-1] != previous:
                    return httpx.Response(409, json={"detail": {"code": "chain_prefix_mismatch", "message": "Full prefix changed"}})
                logical = messages
            reused = len(json.dumps(previous)) // 4
        else:
            logical = messages
        content = logical[-1]["content"]
        if content.startswith("TASK: PLAN"):
            reply = {"constraints": [
                {"id": "identity", "description": "The person directed Iris Observatory in 2007", "time_scope": "2007", "required": True, "answer_target": False},
                {"id": "instrument", "description": "Which instrument this person designed", "time_scope": "unspecified", "required": True, "answer_target": True}],
                "cells": [{"id": "cell1", "focus": "Identify the director and their instrument", "constraint_ids": ["identity", "instrument"], "initial_queries": ["Iris Observatory director 2007"]}]}
            if self.two_cells:
                reply["cells"].append({**deepcopy(reply["cells"][0]), "id": "cell2"})
        elif content.startswith("TASK: DECIDE"):
            data = json.loads(content.rsplit("\n", 1)[-1])
            evidence = data["validated_evidence"]
            targets = [e for e in evidence if e.get("answer_value")]
            reply = {"action": "answer" if targets else "unresolved", "candidate": "Eira Stone",
                     "exact_answer": "Aurora spectrograph" if targets else "",
                     "explanation": "The annual report identifies the director; the design archive names the instrument.",
                     "evidence_ids": [e["evidence_id"] for e in evidence],
                     "reopen_cell_id": None, "next_queries": [], "read_more": []}
        else:
            data = json.loads(content)
            entries = []
            for source in data["source_passages"]:
                if source["docid"] == "demo_identity":
                    entries.append({"candidate": "Eira Stone", "constraint_id": "identity", "time_scope": "2007",
                                    "relation": "SUPPORTS", "source_id": source["source_id"],
                                    "quote": "In 2007, Eira Stone was the director of Iris Observatory.",
                                    "claim": "Eira Stone is the required director."})
                elif source["docid"] == "demo_instrument":
                    entries.append({"candidate": "Eira Stone", "constraint_id": "instrument", "time_scope": "unspecified",
                                    "relation": "SUPPORTS", "source_id": source["source_id"],
                                    "quote": "Eira Stone designed the Aurora spectrograph.",
                                    "claim": "The requested instrument is the Aurora spectrograph.",
                                    "answer_value": "Aurora spectrograph"})
            has_answer = any(e.get("answer_value") for e in entries)
            reply = {"evidence": entries, "retract_ids": [],
                     "next_queries": [] if has_answer else ["Eira Stone designed instrument"],
                     "read_more": [], "summary": "Director identified; instrument checked when available.",
                     }
        # Nontrivial whitespace is intentional: changing raw assistant formatting
        # should make the simulated chain prefix test fail.
        assistant = {"role": "assistant", "content": json.dumps(reply, indent=2) + "\n"}
        response: dict[str, Any] = {"id": "mock", "model": body["model"],
             "choices": [{"message": assistant, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": len(json.dumps(logical)) // 4,
                       "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": reused}}}
        if is_reader and self.chain:
            if chain_id is None:
                self.sequence += 1
                chain_id = f"demo_chain_{self.sequence}"
                status = "created"
            else:
                status = "resumed"
            self.chains[chain_id] = [*deepcopy(logical), deepcopy(assistant)]
            response.update(chain_id=chain_id, chain_status=status)
            response["usage"]["reused_tokens"] = reused
        return httpx.Response(200, json=response)


def make_demo_runtime(store: Store, *, chain: bool = True, evict_once: bool = False,
                      two_cells: bool = False) -> tuple[Runtime, DemoHTTP, httpx.AsyncClient]:
    main = EndpointConfig(base_url="http://mock-main/v1", model="demo-model")
    reader = EndpointConfig(base_url="http://mock-reader/v1", model="demo-model",
                            engine="sparse-vllm" if chain else "vllm",
                            method="h2o" if chain else "vanilla", cache="chain" if chain else "prefix")
    config = AppConfig(main=main, reader=reader,
                       workflow=WorkflowConfig(allow_approximate_tokenizer=True,
                                               max_cells_per_query=2 if two_cells else 1),
                       run_label="SYNTHETIC_PROTOCOL_DEMO")
    backend = DemoHTTP(chain, evict_once, two_cells)
    http = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    clients = {"main": ChatClient(main, store, http=http), "reader": ChatClient(reader, store, http=http)}
    retriever = RecordedRetriever(FixtureRetriever(CORPUS), store)
    return Runtime(config, store, retriever, clients, {"main": Utf8Counter(), "reader": Utf8Counter()}), backend, http


async def drive_nodes_for_test(runtime: Runtime, state: dict) -> dict:
    """Exercise actual node functions without LangGraph; this is a TEST driver only.

    Production 'run' and 'demo' use graphs.build_graph. This helper does not test
    LangGraph Send/barrier/checkpoint semantics and is never reported as doing so.
    """
    state = {**state, **await runtime.plan(state)}
    snapshots = {}
    for job in state["jobs"]:
        cell = runtime.init_cell({**state, "job": job, "previous": None})
        for _ in range(40):
            cell.update(await runtime.cell_fetch(cell))
            if cell["stop_reason"]:
                break
            cell.update(await runtime.cell_read(cell))
            if cell["stop_reason"]:
                break
            cell.update(runtime.cell_validate(cell))
            if cell["stop_reason"]:
                break
        else:
            raise AssertionError("Cell did not terminate")
        snapshots[cell["cell_id"]] = cell
    state["cells"] = snapshots
    state.update(runtime.collect(state))
    state.update(await runtime.decide(state))
    state.update(runtime.render(state))
    return state
