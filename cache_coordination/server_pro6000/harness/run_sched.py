"""Harness-owned scheduling on top of native vLLM (APC + CPU L2 + async restore).

Goal: mean and P95 TTFT no higher than the native-FCFS baseline, mean request
E2E lower. The engine keeps doing what it already does well -- asynchronous KV
restore from the CPU tier, skipping requests whose restore is still in flight.
The harness only changes *which* request enters the engine next and *when*.

Why the queue lives in the harness
    vLLM fixes a request's priority when it is submitted; once it sits in the
    engine queue its position cannot be revised. So the harness holds every
    arrived turn itself and lets at most `k` requests wait inside the engine.
    Those <= k requests are the "next candidates": when one of them is a CPU
    hit, the engine reserves its GPU blocks and starts the async restore as
    soon as blocks are free (it will not admit it otherwise, and will not
    preempt a running request for it -- scheduler.py: "Admit it only if it
    fits in (free - other in-flight reservations)"). That is the prefetch:
    same request, one transfer, no speculative max_tokens=1 warm-up.

Release signal
    in_engine    = released - completed          (exact, harness side)
    waiting_est  = in_engine - num_requests_running   (gauge, one engine step late)
    release while waiting_est < k.
    The gauge lag only over-estimates waiting_est, so it can delay a release by
    one step but never over-release. Re-evaluated on every arrival, every
    completion, and every metrics tick (default 50 ms).

Which request (group `sched`)
    cost(req) = L2 tokens * restore_s_per_token + uncached tokens * prefill_s_per_token
    GPU-resident tokens cost nothing. Lowest cost first, ties by arrival.

Waiting protection (heuristic, not a per-request guarantee)
    budget(req) = baseline P95 TTFT of req's prompt-length bucket, taken from a
    calibration run on a *different* seed and frozen before testing.
    A queued request is urgent when
        waited + ewma(engine queue) + cost(req) >= guard * budget(req)
    Urgent requests are released first, earliest deadline first. Without a
    budget file the guard is off (pure cost order).

Group `tiered`: classify each turn on arrival by where its reusable prefix is.
    L1    reusable part (>= 50% of prompt) is all on the GPU  -> run first
    L2    reusable part partly/wholly on the CPU tier          -> prefetch
    cold  reusable part < 50%                                  -> last
    L1 and L2 turns go to the engine the moment they arrive. The engine ranks
    L1 ahead of L2, so L1 turns compute while an L2 turn, admitted as soon as
    blocks are free, restores asynchronously (the engine skips it until its
    KV lands) -- restore overlaps compute; nothing is predicted.
    Cold turns wait in the harness and enter when the engine holds < k waiting
    requests, FIFO. A cold turn that becomes urgent under the guard is sent at
    once, ahead of every tier, to keep the slowest wait bounded.
    Engine priority (lower first): urgent 0 < L1 1e7 < L2 2e7 < cold 3e7, plus
    release sequence inside a tier.

Group `hfcfs` (control): the harness owns the queue exactly like `sched` --
    at most k requests waiting inside the engine -- but releases strictly in
    arrival order and never consults the oracle. FIFO keeps waits as even as
    possible across requests. It separates "the harness controls admission"
    from "the harness knows where the KV is".

Group `baseline`: every turn released on arrival, priority 0 -> native FCFS.

Every timing starts at user submission (harness arrival), so harness queueing
is inside TTFT. Records keep the run_timeline.py format so the existing plot
and comparison scripts work, plus:
    arr_gpu/arr_cpu    oracle view when the turn arrived (reusable then)
    rel_gpu/rel_cpu    oracle view when it was released to the engine
    harness_wait_s     arrival -> release
    reason             "fcfs" | "cost" | "urgent"
    est_cost_s         the cost estimate used for the decision

usage:
    run_sched.py run --group {baseline,sched} --seed S --out X.json [--budgets B.json] ...
    run_sched.py calibrate --out B.json <baseline_run.json> [...]
"""

import argparse
import json
import queue
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from engine_state import EngineState
from oracle import CacheOracle

T0 = time.perf_counter()
LEN_EDGES = [0, 12000, 18000, 24000, 30000, 10**9]
# engine priority bases for group `tiered` (lower is served first)
TIER_URGENT, TIER_L1, TIER_L2, TIER_COLD = 0, 10**7, 2 * 10**7, 3 * 10**7


def now():
    return time.perf_counter() - T0


def post(url, model, prompt, n_out, priority):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


# ------------------------------------------------------------------ workload
class Conv:
    def __init__(self, inst, traj):
        self.id = inst["id"]
        self.prompts = []
        ids = list(inst["head_ids"])
        for t, turn in enumerate(traj["turns"]):
            self.prompts.append(ids)
            if t < len(inst["turn_suffix_ids"]):
                ids = ids + turn["output_ids"] + inst["turn_suffix_ids"][t]
        self.out_lens = [len(t["output_ids"]) for t in traj["turns"]]


def build_workload(args):
    """Same sampling as run_timeline.py, so a seed reproduces the same users."""
    spec = json.loads(Path(args.instances).read_text())["instances"]
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    rng = random.Random(args.seed)
    picked = rng.sample(spec, args.n)
    convs = [Conv(i, traj[i["id"]]) for i in picked]
    arr = random.Random("%d:arrival" % args.seed)
    arrivals = [min(args.arrival_window, max(0.0, arr.gauss(args.arrival_mean, args.arrival_std)))
                for _ in convs]
    thinks = [[max(args.think_min, random.Random("%d:think:%s:%d" % (args.seed, c.id, r))
                   .gauss(args.think_mean, args.think_std))
               for r in range(len(c.prompts))] for c in convs]
    return convs, arrivals, thinks


class Turn:
    def __init__(self, inst, r, prompt, n_out, think_after):
        self.inst, self.r, self.prompt, self.n_out = inst, r, prompt, n_out
        self.think_after = think_after
        self.t_arrive = self.t_send = self.t_recv = None
        self.queue_ms = self.ttft_ms = self.gen_ms = None
        self.cached = 0
        self.priority = 0
        self.held = 0
        self.arr_gpu = self.arr_cpu = 0
        self.rel_gpu = self.rel_cpu = 0
        self.reason = ""
        self.tier = ""
        self.est_cost_s = 0.0
        self.done_evt = threading.Event()

    def row(self):
        d = {k: getattr(self, k) for k in (
            "inst", "r", "n_out", "think_after", "t_arrive", "t_send", "t_recv",
            "queue_ms", "ttft_ms", "gen_ms", "cached", "priority", "held",
            "arr_gpu", "arr_cpu", "rel_gpu", "rel_cpu", "reason", "tier", "est_cost_s")}
        d["prompt_len"] = len(self.prompt)
        d["pred_gpu"], d["pred_cpu"] = self.rel_gpu, self.rel_cpu     # run_timeline name
        gen = (self.gen_ms or 0) / 1000.0
        ttft = (self.ttft_ms or 0) / 1000.0
        d["decode_start"] = self.t_recv - gen
        d["prefill_start"] = d["decode_start"] - ttft
        d["decode_end"] = self.t_recv
        d["harness_wait_s"] = self.t_send - self.t_arrive
        engine = ((self.queue_ms or 0) + (self.ttft_ms or 0) + (self.gen_ms or 0)) / 1000.0
        d["overhead_s"] = (self.t_recv - self.t_send) - engine
        return d


# ------------------------------------------------------------------ engine view
class GaugePoller:
    """Polls /metrics in the background; plan() reads the latest values."""

    def __init__(self, engine: EngineState, period_s: float):
        self.engine, self.period = engine, period_s
        self.running = 0.0
        self.waiting = 0.0
        self.usage = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            st = self.engine.sample()
            self.running = st.get("running", 0.0)
            self.waiting = st.get("waiting", 0.0)
            self.usage = st.get("usage", 0.0)
            self._stop.wait(self.period)


def load_budgets(path):
    if not path:
        return None
    b = json.loads(Path(path).read_text())
    return b["edges"], b["p95"]


def budget_for(budgets, n_tokens):
    edges, p95 = budgets
    for i in range(len(edges) - 1):
        if edges[i] <= n_tokens < edges[i + 1]:
            return p95[i]
    return p95[-1]


# ------------------------------------------------------------------ graph
class State(TypedDict):
    finished: int
    events: int
    records: Annotated[list, lambda a, b: a + b]
    decisions: Annotated[list, lambda a, b: a + b]


def build(cfg):
    group, url, model = cfg["group"], cfg["url"], cfg["model"]
    oracle, gauges, budgets = cfg["oracle"], cfg["gauges"], cfg["budgets"]
    evq, turns, total = cfg["events"], cfg["turns"], cfg["total"]
    k, guard = cfg["k"], cfg["guard"]
    restore_s, prefill_s = cfg["restore_s_per_token"], cfg["prefill_s_per_token"]
    pool = ThreadPoolExecutor(max_workers=512)
    box = {"Q": [], "in_engine": 0, "seq": 0, "ewma_q": 0.0}

    def cost(t):
        gpu, cpu, total_hit = oracle.match(t.prompt)
        uncached = max(0, len(t.prompt) - total_hit)
        return gpu, cpu, cpu * restore_s + uncached * prefill_s

    def release(tid, reason, est, tier_base=0):
        t = turns[tid]
        t.reason, t.est_cost_s = reason, est
        t.rel_gpu, t.rel_cpu, _ = oracle.match(t.prompt)
        if group == "baseline":
            t.priority = 0
        else:
            box["seq"] += 1
            # tier first, then release order: the engine keeps the harness's order
            t.priority = int(tier_base) + box["seq"]
        box["in_engine"] += 1

        def run():
            t.t_send = now()
            r = post(url, model, t.prompt, t.n_out, t.priority)
            t.t_recv = now()
            m = r.get("metrics") or {}
            t.queue_ms = m.get("queue_time_ms")
            t.ttft_ms = m.get("time_to_first_token_ms")
            t.gen_ms = m.get("generation_time_ms")
            t.cached = ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            evq.put(("done", tid))
        pool.submit(run)

    def plan_tiered(state: State):
        """Q holds cold turns only; L1/L2 turns were sent on arrival."""
        Q = box["Q"]
        tnow = now()
        decisions = []

        def log(tid, reason):
            t = turns[tid]
            decisions.append({"t": tnow, "inst": t.inst, "r": t.r, "reason": reason,
                              "tier": t.tier, "cost": t.est_cost_s, "queue": len(Q),
                              "waited": tnow - t.t_arrive, "in_engine": box["in_engine"],
                              "running": gauges.running})

        # 1. urgent cold turns go now, ahead of every tier and regardless of k
        if budgets is not None:
            for tid in sorted(Q, key=lambda x: turns[x].t_arrive):
                t = turns[tid]
                b = budget_for(budgets, len(t.prompt))
                if (tnow - t.t_arrive) + box["ewma_q"] + t.est_cost_s >= guard * b:
                    Q.remove(tid)
                    release(tid, "urgent", t.est_cost_s, TIER_URGENT)
                    log(tid, "urgent")
        # 2. other cold turns, oldest first, only while the engine has room
        while Q and (box["in_engine"] - gauges.running) < k:
            tid = min(Q, key=lambda x: turns[x].t_arrive)
            Q.remove(tid)
            release(tid, "cold", turns[tid].est_cost_s, TIER_COLD)
            log(tid, "cold")
        return {"events": state["events"] + 1, "decisions": decisions}

    def plan(state: State):
        if group == "tiered":
            return plan_tiered(state)
        Q = box["Q"]
        if not Q:
            return {"events": state["events"] + 1}
        if group == "hfcfs":
            # control: same admission gate as `sched`, arrival order, no cache view
            decisions = []
            tnow = now()
            while Q and (box["in_engine"] - gauges.running) < k:
                tid = min(Q, key=lambda x: turns[x].t_arrive)
                Q.remove(tid)
                release(tid, "fifo", 0.0)
                decisions.append({"t": tnow, "inst": turns[tid].inst, "r": turns[tid].r,
                                  "reason": "fifo", "queue": len(Q),
                                  "waited": tnow - turns[tid].t_arrive,
                                  "in_engine": box["in_engine"], "running": gauges.running})
            return {"events": state["events"] + 1, "decisions": decisions}
        if group == "baseline":
            for tid in Q:
                release(tid, "fcfs", 0.0)
            box["Q"] = []
            return {"events": state["events"] + 1}

        room = k - (box["in_engine"] - gauges.running)
        if room < 1:                          # engine already holds k waiting requests
            return {"events": state["events"] + 1}

        # score once per decision point: oracle lookups walk ~2k blocks each and
        # share a lock with the event-ingest thread
        tnow = now()
        scored = []
        for tid in Q:
            t = turns[tid]
            _, _, c = cost(t)
            urgent, deadline = False, float("inf")
            if budgets is not None:
                b = budget_for(budgets, len(t.prompt))
                deadline = t.t_arrive + guard * b
                urgent = (tnow - t.t_arrive) + box["ewma_q"] + c >= guard * b
            scored.append((tid, c, urgent, deadline))
        n_urgent = sum(1 for s in scored if s[2])
        # urgent first (earliest deadline), then cheapest; ties by arrival
        scored.sort(key=lambda s: ((0, s[3]) if s[2] else (1, s[1]), turns[s[0]].t_arrive))

        decisions = []
        for tid, c, urgent, _ in scored[:int(room)]:
            Q.remove(tid)
            release(tid, "urgent" if urgent else "cost", c)
            decisions.append({"t": tnow, "inst": turns[tid].inst, "r": turns[tid].r,
                              "reason": "urgent" if urgent else "cost", "cost": c,
                              "queue": len(Q), "in_engine": box["in_engine"],
                              "running": gauges.running, "n_urgent": n_urgent})
        return {"events": state["events"] + 1, "decisions": decisions}

    def dispatch(state: State):
        return {}

    def classify_and_route(tid):
        """tiered: L1/L2 turns go to the engine now; cold ones queue in the harness."""
        t = turns[tid]
        n = len(t.prompt)
        hit = t.arr_gpu + t.arr_cpu
        t.est_cost_s = t.arr_cpu * restore_s + max(0, n - hit) * prefill_s
        if hit >= 0.5 * n:
            t.tier = "L1" if t.arr_gpu >= hit - 256 else "L2"
            release(tid, t.tier, t.est_cost_s, TIER_L1 if t.tier == "L1" else TIER_L2)
            return {"t": t.t_arrive, "inst": t.inst, "r": t.r, "reason": t.tier,
                    "tier": t.tier, "cost": t.est_cost_s, "queue": len(box["Q"]),
                    "waited": 0.0, "in_engine": box["in_engine"], "running": gauges.running}
        t.tier = "cold"
        box["Q"].append(tid)
        return None

    def collect(state: State):
        recs, finished = [], state["finished"]
        decisions = []
        items = []
        try:
            items.append(evq.get(timeout=cfg["tick_s"]))
        except queue.Empty:
            pass
        while True:
            try:
                items.append(evq.get_nowait())
            except queue.Empty:
                break
        for kind, tid in items:
            t = turns[tid]
            if kind == "arrive":
                t.arr_gpu, t.arr_cpu, _ = oracle.match(t.prompt)
                if group == "tiered":
                    dec = classify_and_route(tid)
                    if dec:
                        decisions.append(dec)
                else:
                    box["Q"].append(tid)
            else:
                box["in_engine"] -= 1
                if t.queue_ms is not None:
                    box["ewma_q"] = 0.8 * box["ewma_q"] + 0.2 * t.queue_ms / 1000.0
                recs.append(t.row())
                finished += 1
                t.done_evt.set()
        return {"finished": finished, "records": recs, "decisions": decisions}

    def route(state: State):
        return END if state["finished"] >= total else "plan"

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("dispatch", dispatch)
    g.add_node("collect", collect)
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "collect")
    g.add_conditional_edges("collect", route, {"plan": "plan", END: END})
    return g.compile()


# ------------------------------------------------------------------ commands
def cmd_run(args):
    convs, arrivals, thinks = build_workload(args)
    total = sum(len(c.prompts) for c in convs)
    budgets = load_budgets(args.budgets)

    oracle = CacheOracle(args.events).start()
    engine = EngineState(args.base)
    gauges = GaugePoller(engine, args.tick_s).start()
    time.sleep(1.0)
    root = args.base.rstrip("/").rsplit("/v1", 1)[0]
    urllib.request.urlopen(urllib.request.Request(
        root + "/reset_prefix_cache", data=b"",
        headers={"Content-Type": "application/json"}), timeout=60).read()
    time.sleep(2.0)

    evq: queue.Queue = queue.Queue()
    turns: dict = {}
    graph = build({
        "group": args.group, "url": args.base.rstrip("/") + "/completions",
        "model": args.model, "oracle": oracle, "gauges": gauges, "budgets": budgets,
        "events": evq, "turns": turns, "total": total, "k": args.k, "guard": args.guard,
        "restore_s_per_token": args.restore_us_per_token / 1e6,
        "prefill_s_per_token": args.prefill_us_per_token / 1e6,
        "tick_s": args.tick_s,
    })

    t_zero = [None]

    def user(i):
        c = convs[i]
        time.sleep(max(0.0, t_zero[0] + arrivals[i] - now()))
        for r in range(len(c.prompts)):
            t = Turn(i, r, c.prompts[r], c.out_lens[r], thinks[i][r])
            tid = id(t)
            turns[tid] = t
            t.t_arrive = now()
            evq.put(("arrive", tid))
            t.done_evt.wait()
            if r + 1 < len(c.prompts):
                time.sleep(thinks[i][r])

    print("group=%s seed=%d instances=%d turns=%d k=%d guard=%s budgets=%s"
          % (args.group, args.seed, len(convs), total, args.k,
             args.guard if budgets else "off", args.budgets or "-"))
    t_zero[0] = now()
    threads = [threading.Thread(target=user, args=(i,), daemon=True) for i in range(len(convs))]
    for th in threads:
        th.start()
    final = graph.invoke({"finished": 0, "events": 0, "records": [], "decisions": []},
                         {"recursion_limit": 10_000_000})
    gauges.stop()
    oracle.stop()

    R = final["records"]
    for d in R:
        for key in ("t_arrive", "t_send", "t_recv", "prefill_start", "decode_start", "decode_end"):
            d[key] -= t_zero[0]
    for d in final["decisions"]:
        d["t"] -= t_zero[0]
    summarize(R)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "group": args.group, "n": args.n, "seed": args.seed,
        "arrival": [args.arrival_mean, args.arrival_std, args.arrival_window],
        "think": [args.think_mean, args.think_std, args.think_min],
        "k": args.k, "guard": args.guard if budgets else None, "budgets": args.budgets,
        "restore_us_per_token": args.restore_us_per_token,
        "prefill_us_per_token": args.prefill_us_per_token,
        "tier_label": args.tier_label,
        "records": R, "decisions": final["decisions"],
    }))
    print("wrote %s" % args.out)


def summarize(R):
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]
    ttft = lambda d: max(d["decode_start"], d["t_arrive"]) - d["t_arrive"]
    first = [d for d in R if d["r"] == 0]
    later = [d for d in R if d["r"] > 0]
    mean = lambda xs: sum(xs) / len(xs)
    for lab, S in (("all", R), ("first", first), ("follow-up", later)):
        xs = [ttft(d) for d in S]
        print("TTFT %-9s mean %6.1f  p50 %6.1f  p95 %6.1f  p99 %6.1f  max %6.1f"
              % (lab, mean(xs), q(xs, .5), q(xs, .95), q(xs, .99), max(xs)))
    e2e = [d["decode_end"] - d["t_arrive"] for d in R]
    print("E2E  all       mean %6.1f  p50 %6.1f  p95 %6.1f" % (mean(e2e), q(e2e, .5), q(e2e, .95)))
    hw = [d["harness_wait_s"] for d in R]
    print("harness wait   mean %6.2f  p95 %6.2f  max %6.2f" % (mean(hw), q(hw, .95), max(hw)))
    local = sum(d["prompt_len"] - d["cached"] for d in R)
    lost = sum(max(0, d["arr_gpu"] + d["arr_cpu"] - d["cached"]) for d in R)
    print("local prompt compute %.2fM tokens   reusable-at-arrival but lost before execution %.2fM"
          % (local / 1e6, lost / 1e6))
    print("makespan %.1fs over %d turns" % (max(d["decode_end"] for d in R), len(R)))
    tiers = sorted(set(d.get("tier", "") for d in R) - {""})
    for tr in tiers:
        S = [d for d in R if d.get("tier") == tr]
        xs = [ttft(d) for d in S]
        print("  tier %-5s n=%3d  TTFT mean %6.1f  max %6.1f   (urgent releases: %d)"
              % (tr, len(S), mean(xs), max(xs), sum(1 for d in S if d.get("reason") == "urgent")))


def cmd_calibrate(args):
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]
    xs_by = [[] for _ in range(len(LEN_EDGES) - 1)]
    for path in args.runs:
        for d in json.loads(Path(path).read_text())["records"]:
            tt = max(d["decode_start"], d["t_arrive"]) - d["t_arrive"]
            for i in range(len(LEN_EDGES) - 1):
                if LEN_EDGES[i] <= d["prompt_len"] < LEN_EDGES[i + 1]:
                    xs_by[i].append(tt)
    everything = [x for xs in xs_by for x in xs]
    p95 = [q(xs, .95) if len(xs) >= args.min_samples else q(everything, .95) for xs in xs_by]
    out = {"edges": LEN_EDGES, "p95": p95, "n": [len(xs) for xs in xs_by],
           "overall_p95": q(everything, .95), "source": args.runs}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for i in range(len(LEN_EDGES) - 1):
        print("prompt %6d-%-10d n=%4d  baseline P95 TTFT %.1fs%s" % (
            LEN_EDGES[i], LEN_EDGES[i + 1], len(xs_by[i]), p95[i],
            "" if len(xs_by[i]) >= args.min_samples else "  (few samples: overall P95)"))
    print("wrote %s" % args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--group", required=True, choices=["baseline", "hfcfs", "sched", "tiered"])
    r.add_argument("--instances", default="data/instances_320.json")
    r.add_argument("--traj", default="trajectories/trajectories_320.json")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--arrival-mean", type=float, default=100.0)
    r.add_argument("--arrival-std", type=float, default=40.0)
    r.add_argument("--arrival-window", type=float, default=200.0)
    r.add_argument("--think-mean", type=float, default=5.0)
    r.add_argument("--think-std", type=float, default=1.0)
    r.add_argument("--think-min", type=float, default=0.5)
    r.add_argument("--base", default="http://127.0.0.1:19081/v1")
    r.add_argument("--model", default="Qwen3-8B")
    r.add_argument("--events", default="tcp://127.0.0.1:5557")
    r.add_argument("--k", type=int, default=2, help="max requests waiting inside the engine")
    r.add_argument("--budgets", default=None, help="frozen calibration file; omit to disable the guard")
    r.add_argument("--guard", type=float, default=1.0, help="urgent when predicted TTFT >= guard * budget")
    r.add_argument("--restore-us-per-token", type=float, default=3.05,
                   help="L2->L1 restore cost (measured 48.4 GB/s at 144 KiB/token)")
    r.add_argument("--prefill-us-per-token", type=float, default=110.0,
                   help="prefill cost under load (measured ~9.1k tok/s)")
    r.add_argument("--tick-s", type=float, default=0.05)
    r.add_argument("--tier-label", default="L1 + L2 96 GiB")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("calibrate")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", required=True)
    c.add_argument("--min-samples", type=int, default=20)
    c.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
