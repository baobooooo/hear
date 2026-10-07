"""Replay the recorded SCBench trajectories through vLLM under several schedules.

Every arm sends exactly the same requests with exactly the same prompt token ids
and the same forced output length (the recorded answers are teacher-forced,
min_tokens == max_tokens == len(recorded), ignore_eos). Arms differ only in
which requests the harness releases, in what order, and with what priority.

    baseline  release everything at once, no priority. What a plain LangGraph
              batch over the conversations does.
    capped    fixed slot count, still turn-major. Isolates "just pick a sane
              concurrency number".
    sticky    fixed slot count, conversation-major. Isolates "run a conversation
              to the end before starting another", without engine feedback.
    kvaware   sticky plus oracle ranking (and, with --priority, that ranking is
              handed to the engine instead of merely implied by send order).
    adaptive  no fixed slot count at all. Each round the harness asks the oracle
              how much of every candidate the engine already holds, so a
              candidate's real cost is prompt_len - hit; it then admits by token
              budget against the live KV pool occupancy. With mixed prefix
              lengths there is no single right concurrency, so this is the only
              arm whose wave size can be right.

The LangGraph is one graph over the whole batch, not one per conversation:

    START -> plan -> dispatch -> collect -> (work left ? plan : END)
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from engine_state import EngineState
from oracle import CacheOracle


# --------------------------------------------------------------------- engine
def post(url, model, prompt, n_out, priority=0):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": n_out,
        "min_tokens": n_out,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": False,
        "priority": priority,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        r = json.loads(resp.read())
    dt = time.perf_counter() - t0
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return dt, u["prompt_tokens"], cached


# ----------------------------------------------------------------- work items
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
        self.waits = 0

    @property
    def done(self):
        return self.turn >= len(self.prompts)

    def next_prompt(self):
        return self.prompts[self.turn]


# ---------------------------------------------------------------- graph state
class State(TypedDict):
    active: list
    pending: list
    done: list
    rounds: int
    records: Annotated[list, lambda a, b: a + b]
    waves: Annotated[list, lambda a, b: a + b]


def build_graph(cfg):
    convs, oracle, engine = cfg["convs"], cfg["oracle"], cfg["engine"]
    url, model, arm = cfg["url"], cfg["model"], cfg["arm"]
    slots, token_budget = cfg["slots"], cfg["token_budget"]
    stagger_s = cfg["stagger_ms"] / 1000.0
    use_priority = cfg["use_priority"]
    capacity = cfg["capacity"]
    max_seqs = cfg["max_seqs"]
    state_box = {"alpha": cfg["alpha"], "preemptions": 0.0, "wave": None}

    def measure(cid):
        p = convs[cid].next_prompt()
        gpu, cpu, _ = oracle.match(p)
        return p, gpu, cpu

    def plan_adaptive(active, pending):
        st = engine.sample()
        alpha = state_box["alpha"]
        if st["preemptions"] > state_box["preemptions"]:
            alpha = max(0.35, alpha * 0.85)      # engine is thrashing, back off
        else:
            alpha = min(0.90, alpha * 1.10)
        state_box["preemptions"] = st["preemptions"]
        state_box["alpha"] = alpha

        # A hit saves compute, not memory: every block of an admitted request
        # must be resident for it to run. So the budget is total residency.
        budget = alpha * capacity
        if st["usage"] > 0.95:
            budget = min(budget, 0.8 * capacity)   # engine says the pool is full

        held = list(active)                        # sticky: keep them to the end
        scored = []
        for cid in pending:
            p, gpu, cpu = measure(cid)
            scored.append({"cid": cid, "len": len(p), "gpu": gpu, "cpu": cpu,
                           "need": len(p) - gpu, "waits": convs[cid].waits})
        # fill free budget with warm conversations first (their prefix is
        # perishable), then with short ones, which leave room for more
        scored.sort(key=lambda s: (-s["gpu"], s["len"]))

        admitted = list(held)
        used = sum(len(convs[c].next_prompt()) for c in held)
        rest = []
        starved = max((s for s in scored if s["waits"] >= cfg["starve_rounds"]),
                      key=lambda s: s["waits"], default=None)
        if starved is not None:            # never let a long cold one starve
            admitted.append(starved["cid"])
            used += starved["len"]
        for s in scored:
            if s["cid"] in admitted:
                continue
            if len(admitted) < max_seqs and used + s["len"] <= budget:
                admitted.append(s["cid"])
                used += s["len"]
            else:
                rest.append(s["cid"])
        wave = {"admitted": len(admitted), "held": len(held),
                "pending": len(rest), "alpha": round(alpha, 3),
                "budget": int(budget), "prompt_tokens": used,
                "kv_usage": st["usage"], "preemptions": st["preemptions"],
                "waiting_capacity": st.get("waiting_capacity", 0.0)}
        for cid in rest:
            convs[cid].waits += 1
        for cid in admitted:
            convs[cid].waits = 0
        return admitted, rest, wave

    def plan(state: State):
        active = [c for c in state["active"] if not convs[c].done]
        pending = [c for c in state["pending"] if not convs[c].done]
        rounds = state["rounds"] + 1

        if arm == "baseline":
            allc = active + pending
            return {"active": allc, "pending": [], "rounds": rounds,
                    "waves": [{"admitted": len(allc), "pending": 0}]}

        if arm == "adaptive":
            admitted, rest, wave = plan_adaptive(active, pending)
            return {"active": admitted, "pending": rest, "rounds": rounds,
                    "waves": [wave]}

        sticky = arm in ("sticky", "kvaware")
        if sticky:
            held, cands = list(active), list(pending)
        else:
            held, cands = [], sorted(active + pending,
                                     key=lambda c: (convs[c].turn, c))
        if arm == "kvaware":
            cands = [s[1] for s in sorted(
                (-measure(c)[1], c) for c in cands)]
        elif sticky:
            cands = sorted(cands)

        used = sum(len(convs[c].next_prompt()) for c in held)
        admitted, rest = list(held), []
        for cid in cands:
            need = len(convs[cid].next_prompt())
            if len(admitted) < slots and used + need <= token_budget:
                admitted.append(cid)
                used += need
            else:
                rest.append(cid)
        return {"active": admitted, "pending": rest, "rounds": rounds,
                "waves": [{"admitted": len(admitted), "pending": len(rest),
                           "prompt_tokens": used}]}

    def dispatch(state: State):
        if not state["active"]:
            return {}
        jobs = []
        for cid in state["active"]:
            c = convs[cid]
            p, gpu, cpu = measure(cid)
            prio = -((gpu + cpu) // 16) if use_priority else 0
            jobs.append((cid, c.turn, p, c.out_lens[c.turn], gpu, cpu, prio))

        def run(item):
            idx, (cid, turn, p, n_out, gpu, cpu, prio) = item
            if stagger_s:
                time.sleep(idx * stagger_s)
            dt, plen, cached = post(url, model, p, n_out, prio)
            return {"conv": cid, "turn": turn, "prompt_len": plen,
                    "cached_tokens": cached, "latency_s": dt,
                    "pred_gpu": gpu, "pred_cpu": cpu, "priority": prio,
                    "n_out": n_out}

        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            recs = list(ex.map(run, list(enumerate(jobs))))
        return {"records": recs}

    def collect(state: State):
        for cid in state["active"]:
            convs[cid].turn += 1
        active = [c for c in state["active"] if not convs[c].done]
        finished = [c for c in state["active"] if convs[c].done]
        return {"active": active, "done": state["done"] + finished}

    def route(state: State):
        return "plan" if (state["active"] or state["pending"]) else END

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("dispatch", dispatch)
    g.add_node("collect", collect)
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "collect")
    g.add_conditional_edges("collect", route, {"plan": "plan", END: END})
    return g.compile()


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "capped", "sticky", "kvaware", "adaptive"])
    ap.add_argument("--instances", default="data/instances_het.json")
    ap.add_argument("--traj", default="trajectories/trajectories_het.json")
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--events", default="tcp://127.0.0.1:5557")
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--token-budget", type=int, default=400000)
    ap.add_argument("--capacity", type=int, default=518080,
                    help="engine GPU KV pool, in tokens (from its startup log)")
    ap.add_argument("--max-seqs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--starve-rounds", type=int, default=6)
    ap.add_argument("--stagger-ms", type=float, default=5.0)
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    convs = {i["id"]: Conv(i, traj[i["id"]]) for i in spec["instances"]}
    total_reqs = sum(len(c.prompts) for c in convs.values())

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
        "slots": args.slots, "token_budget": args.token_budget,
        "stagger_ms": args.stagger_ms, "use_priority": args.priority,
        "capacity": args.capacity, "max_seqs": args.max_seqs,
        "alpha": args.alpha, "starve_rounds": args.starve_rounds,
    })

    init: State = {"active": [], "pending": list(convs), "done": [],
                   "rounds": 0, "records": [], "waves": []}
    print("arm=%s conversations=%d requests=%d priority=%s"
          % (args.arm, len(convs), total_reqs, args.priority))
    t0 = time.perf_counter()
    final = graph.invoke(init, {"recursion_limit": 100000})
    wall = time.perf_counter() - t0
    oracle.stop()

    recs = final["records"]
    hit = sum(r["cached_tokens"] for r in recs)
    tot = sum(r["prompt_len"] for r in recs)
    later = [r for r in recs if r["turn"] > 0]
    lhit = sum(r["cached_tokens"] for r in later) or 0
    ltot = sum(r["prompt_len"] for r in later) or 1
    sizes = [w["admitted"] for w in final["waves"]]
    print("\nwall %.1fs over %d requests in %d rounds"
          % (wall, len(recs), final["rounds"]))
    print("prefix cache hit: %.1f%% of all prompt tokens, %.1f%% on turns 2+"
          % (100 * hit / tot, 100 * lhit / ltot))
    print("wave size: min=%d med=%d max=%d  (mean %.1f)"
          % (min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes),
             sum(sizes) / len(sizes)))
    print("oracle: %s" % oracle.snapshot())

    out = args.out or "runs/replay-%s.json" % args.arm
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "arm": args.arm, "wall_s": wall, "rounds": final["rounds"],
        "priority": args.priority, "slots": args.slots,
        "capacity": args.capacity, "records": recs, "waves": final["waves"],
    }))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
