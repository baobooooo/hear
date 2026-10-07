"""Replay recorded SCBench trajectories through vLLM under several schedules.

Dispatch is continuous, not round-based. An earlier version waited for every
request in a wave before planning the next one, so a short request that finished
early just idled until the slowest one returned -- pure waste, and worse the
more uneven the batch. Here the graph loops on completion events: `collect`
waits for the FIRST in-flight request to finish, and whatever budget that frees
is refilled immediately by the next `plan`.

    START -> plan -> dispatch -> collect -> (work left ? plan : END)

with `plan` admitting into whatever budget is free right now, `dispatch`
submitting without blocking, and `collect` harvesting completions.

Arms differ only in how much they let in flight and in what order:

    baseline  no admission control. Every conversation is live at once and its
              next turn is submitted the moment the previous one returns. What a
              harness that ignores the KV pool would do.
    static    at most N conversations in flight; a conversation keeps its slot
              until its last turn (stickiness). N has to be guessed.
    adaptive  no slot count. The oracle says how much of each candidate the
              engine still holds; admission is budgeted in resident tokens
              against the live KV pool, and the ranking is handed to the engine
              as request priority. With mixed prefix lengths there is no single
              correct N, which is the point.

Every arm sends the same requests with the same prompt token ids and the same
forced output lengths (recorded answers, teacher-forced), so only the schedule
differs.
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from engine_state import EngineState
from oracle import CacheOracle


def post(url, model, prompt, n_out, priority=0):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        r = json.loads(resp.read())
    dt = time.perf_counter() - t0
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return dt, u["prompt_tokens"], cached


class Conv:
    """One SCBench conversation with its per-turn prompts precomputed."""

    def __init__(self, inst, traj):
        self.id = inst["id"]
        self.prompts = []
        ids = list(inst["head_ids"])
        for t, turn in enumerate(traj["turns"]):
            self.prompts.append(ids)
            if t < len(inst["turn_suffix_ids"]):
                ids = ids + turn["output_ids"] + inst["turn_suffix_ids"][t]
        self.out_lens = [len(t["output_ids"]) for t in traj["turns"]]
        self.turn = 0

    @property
    def done(self):
        return self.turn >= len(self.prompts)

    def next_prompt(self):
        return self.prompts[self.turn]


class State(TypedDict):
    admitted: list          # conversations holding budget
    queued: list            # waiting for budget
    submit: list            # admitted this step, to be dispatched
    events: int
    records: Annotated[list, lambda a, b: a + b]
    samples: Annotated[list, lambda a, b: a + b]


def build_graph(cfg):
    convs, oracle, engine = cfg["convs"], cfg["oracle"], cfg["engine"]
    url, model, arm = cfg["url"], cfg["model"], cfg["arm"]
    slots, capacity, max_seqs = cfg["slots"], cfg["capacity"], cfg["max_seqs"]
    use_priority = cfg["use_priority"]

    pool = ThreadPoolExecutor(max_workers=max_seqs + 8)
    box = {"inflight": {}, "alpha": cfg["alpha"], "preemptions": 0.0}

    def cost(cid):
        return len(convs[cid].next_prompt())

    def plan(state: State):
        admitted = [c for c in state["admitted"] if not convs[c].done]
        queued = [c for c in state["queued"] if not convs[c].done]
        sample = {"t": time.perf_counter(), "inflight": len(box["inflight"])}

        if arm == "baseline":
            new = queued
            queued = []
        else:
            if arm == "adaptive":
                st = engine.sample()
                alpha = box["alpha"]
                alpha = (max(0.35, alpha * 0.85)
                         if st["preemptions"] > box["preemptions"]
                         else min(0.90, alpha * 1.10))
                box["preemptions"] = st["preemptions"]
                box["alpha"] = alpha
                budget = alpha * capacity
                if st["usage"] > 0.95:
                    budget = min(budget, 0.80 * capacity)
                cap_n = max_seqs
                sample.update(alpha=round(alpha, 3), budget=int(budget),
                              kv_usage=st["usage"], preemptions=st["preemptions"])
                # a hit saves compute, not memory: an admitted request needs all
                # of its blocks resident, so warm ones cost their full length too
                scored = sorted(
                    ((-oracle.match(convs[c].next_prompt())[0], cost(c), c)
                     for c in queued))
                order = [s[2] for s in scored]
            else:
                budget, cap_n = float("inf"), slots
                order = sorted(queued)

            used = sum(cost(c) for c in admitted)
            new, rest = [], []
            for cid in order:
                if len(admitted) + len(new) < cap_n and used + cost(cid) <= budget:
                    new.append(cid)
                    used += cost(cid)
                else:
                    rest.append(cid)
            queued = rest

        sample["admitted"] = len(admitted) + len(new)
        return {"admitted": admitted + new, "queued": queued, "submit": new,
                "events": state["events"] + 1, "samples": [sample]}

    def submit_turn(cid):
        c = convs[cid]
        p = c.next_prompt()
        gpu, cpu = oracle.match(p)[0], oracle.match(p)[1]
        prio = -((gpu + cpu) // 16) if use_priority else 0
        turn = c.turn

        def run():
            dt, plen, cached = post(url, model, p, c.out_lens[turn], prio)
            return {"conv": cid, "turn": turn, "prompt_len": plen,
                    "cached_tokens": cached, "latency_s": dt,
                    "pred_gpu": gpu, "pred_cpu": cpu, "priority": prio}

        box["inflight"][pool.submit(run)] = cid

    def dispatch(state: State):
        for cid in state["submit"]:
            submit_turn(cid)
        return {"submit": []}

    def collect(state: State):
        if not box["inflight"]:
            return {}
        done, _ = wait(list(box["inflight"]), return_when=FIRST_COMPLETED)
        recs = []
        for fut in done:
            cid = box["inflight"].pop(fut)
            recs.append(fut.result())
            convs[cid].turn += 1
            # stickiness: a conversation that still has turns keeps its budget
            # and goes straight back out, without waiting for anyone else
            if not convs[cid].done:
                submit_turn(cid)
        return {"records": recs}

    def route(state: State):
        live = box["inflight"] or state["queued"] or any(
            not convs[c].done for c in state["admitted"])
        return "plan" if live else END

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("dispatch", dispatch)
    g.add_node("collect", collect)
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "collect")
    g.add_conditional_edges("collect", route, {"plan": "plan", END: END})
    return g.compile()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "static", "adaptive"])
    ap.add_argument("--instances", default="data/instances_100.json")
    ap.add_argument("--traj", default="trajectories/trajectories_100.json")
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--events", default="tcp://127.0.0.1:5557")
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--capacity", type=int, default=518080,
                    help="engine GPU KV pool in tokens, from its startup log")
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    convs = {i["id"]: Conv(i, traj[i["id"]]) for i in spec["instances"]}
    n_req = sum(len(c.prompts) for c in convs.values())
    workset = sum(len(c.prompts[-1]) for c in convs.values())

    oracle = CacheOracle(args.events).start()
    engine = EngineState(args.base)
    time.sleep(1.0)
    root = args.base.rstrip("/").rsplit("/v1", 1)[0]
    urllib.request.urlopen(urllib.request.Request(
        root + "/reset_prefix_cache", data=b"",
        headers={"Content-Type": "application/json"}), timeout=60).read()
    time.sleep(2.0)

    graph = build_graph({
        "convs": convs, "oracle": oracle, "engine": engine, "arm": args.arm,
        "url": args.base.rstrip("/") + "/completions", "model": args.model,
        "slots": args.slots, "capacity": args.capacity,
        "max_seqs": args.max_seqs, "alpha": args.alpha,
        "use_priority": args.priority,
    })

    init: State = {"admitted": [], "queued": list(convs), "submit": [],
                   "events": 0, "records": [], "samples": []}
    print("arm=%s conversations=%d requests=%d slots=%s priority=%s"
          % (args.arm, len(convs), n_req, args.slots, args.priority))
    print("working set %d tokens vs pool %d tokens (%.0f%% coverage)"
          % (workset, args.capacity, 100 * args.capacity / workset))

    t0 = time.perf_counter()
    final = graph.invoke(init, {"recursion_limit": 1000000})
    wall = time.perf_counter() - t0
    oracle.stop()

    recs = final["records"]
    hit = sum(r["cached_tokens"] for r in recs)
    tot = sum(r["prompt_len"] for r in recs)
    later = [r for r in recs if r["turn"] > 0]
    lhit = sum(r["cached_tokens"] for r in later)
    ltot = max(1, sum(r["prompt_len"] for r in later))
    infl = [s["inflight"] for s in final["samples"]]
    lat = sorted(r["latency_s"] for r in recs)

    print("\nwall %.1fs over %d requests (%d scheduling events)"
          % (wall, len(recs), final["events"]))
    print("prefix cache hit : %.1f%% of all prompt tokens, %.1f%% on turns 2+"
          % (100 * hit / tot, 100 * lhit / ltot))
    print("recomputed       : %.2fM prompt tokens of %.2fM"
          % ((tot - hit) / 1e6, tot / 1e6))
    print("in flight        : min=%d med=%d max=%d"
          % (min(infl), sorted(infl)[len(infl) // 2], max(infl)))
    print("latency p50/p99  : %.2fs / %.2fs" % (lat[len(lat) // 2], lat[-1]))
    print("oracle           : %s" % oracle.snapshot())

    out = args.out or "runs/replay-%s.json" % args.arm
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "arm": args.arm, "wall_s": wall, "events": final["events"],
        "slots": args.slots, "capacity": args.capacity,
        "priority": args.priority, "records": recs,
        "samples": final["samples"],
    }))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
