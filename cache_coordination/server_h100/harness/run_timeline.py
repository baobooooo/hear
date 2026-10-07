"""Interactive-users replay that records, for every turn, the four phases a
user experiences, so they can be drawn as a timeline and concurrency can be
read off any vertical slice:

    wait      turn arrives -> engine starts working on it (harness hold +
              HTTP + tokenization + engine queue)
    prefill   engine scheduled it -> first token        (engine timestamp)
    decode    first token -> last token                 (engine timestamp)
    thinking  answer done -> this instance's next turn arrives

Workload: N instances sampled from the SCBench set. Instance i's first turn
arrives at A_i ~ Normal(mean, std) clipped to [0, window]; after each answer
the user thinks Z ~ Normal(think_mean, think_std) clipped to >= think_min,
then sends the next turn:

    t_arrive[i, r+1] = t_done[i, r] + Z[i, r]

A_i and Z are pre-sampled from fixed seeds so every group sees the same users.

The engine must run with --enable-per-request-metrics so each response carries
queue_time_ms / time_to_first_token_ms / generation_time_ms. Those are engine
durations; we anchor them at the client receive time to get absolute spans.

Groups:
    baseline  send each turn the moment it arrives, priority 0.
    kvaware   warm turn (prefix still resident) jumps the queue with high
              priority; cold turn is held while the pool is committed.

Harness is one LangGraph over the whole run:
    START -> plan -> dispatch -> collect -> (turns left ? plan : END)
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
from sampler import Sampler

T0 = time.perf_counter()


def now():
    return time.perf_counter() - T0


def post(url, model, prompt, n_out, priority=0, xargs=None):
    body = {
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }
    if xargs:
        body["vllm_xargs"] = xargs
    body = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def reset_all_caches(root, base, model, probe_prompt):
    """Clear L1 + L2, prove it with a probe, clear again. Raises if not clean."""
    def reset():
        for _ in range(30):
            with urllib.request.urlopen(urllib.request.Request(
                    root + "/reset_prefix_cache?reset_external=true", data=b"",
                    headers={"Content-Type": "application/json"}), timeout=60) as r:
                if r.status == 200:
                    return
            time.sleep(1.0)
        raise RuntimeError("reset_prefix_cache did not succeed")
    reset()
    time.sleep(2.0)
    r = post(base.rstrip("/") + "/completions", model, probe_prompt, 1)
    cached = ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if cached > 256:
        raise RuntimeError("cache not empty after reset: probe hit %d tokens" % cached)
    reset()
    time.sleep(2.0)
    print("cache reset verified (L1+L2): probe cached %d tokens" % cached, flush=True)


class Conv:
    def __init__(self, inst, traj):
        self.id = inst["id"]
        self.prompts, self.history = [], []
        ids = list(inst["head_ids"])
        for t, turn in enumerate(traj["turns"]):
            self.prompts.append(ids)
            self.history.append(ids + turn["output_ids"])
            if t < len(inst["turn_suffix_ids"]):
                ids = ids + turn["output_ids"] + inst["turn_suffix_ids"][t]
        self.out_lens = [len(t["output_ids"]) for t in traj["turns"]]


class Turn:
    def __init__(self, inst, r, prompt, n_out, think_after):
        self.inst, self.r, self.prompt, self.n_out = inst, r, prompt, n_out
        self.think_after = think_after
        self.t_arrive = self.t_send = self.t_recv = None
        self.queue_ms = self.ttft_ms = self.gen_ms = None
        self.cached = 0
        self.priority = 0
        self.held = 0
        self.pred_gpu = self.pred_cpu = 0
        self.kvp = None
        self.done_evt = threading.Event()

    def row(self):
        extra = {"kvp": self.kvp} if self.kvp else {}
        d = {k: getattr(self, k) for k in (
            "inst", "r", "n_out", "think_after", "t_arrive", "t_send", "t_recv",
            "queue_ms", "ttft_ms", "gen_ms", "cached", "priority", "held",
            "pred_gpu", "pred_cpu")}
        d["prompt_len"] = len(self.prompt)
        # absolute spans, anchored at the client receive time
        gen = (self.gen_ms or 0) / 1000.0
        ttft = (self.ttft_ms or 0) / 1000.0
        d["decode_start"] = self.t_recv - gen
        d["prefill_start"] = d["decode_start"] - ttft
        d["decode_end"] = self.t_recv
        # client-observed minus engine-accounted: HTTP + tokenize + detokenize
        engine = ((self.queue_ms or 0) + (self.ttft_ms or 0) + (self.gen_ms or 0)) / 1000.0
        d["overhead_s"] = (self.t_recv - self.t_send) - engine
        d.update(extra)
        return d


class State(TypedDict):
    pending: list
    finished: int
    events: int
    records: Annotated[list, lambda a, b: a + b]


def build(cfg):
    oracle, engine = cfg["oracle"], cfg["engine"]
    url, model, group = cfg["url"], cfg["model"], cfg["group"]
    capacity, alpha, total = cfg["capacity"], cfg["alpha"], cfg["total"]
    evq, turns = cfg["events"], cfg["turns"]
    convs, pf_log = cfg["convs"], cfg["pf_log"]
    pf_delay, pf_prio = cfg["pf_delay"], cfg["pf_prio"]
    pool = ThreadPoolExecutor(max_workers=512)
    inflight = set()
    pf_due = {}          # inst -> (due time, next turn index)

    def prefetch(inst, r_next):
        hist = convs[inst].history[r_next - 1]
        gpu, cpu, _ = oracle.match(hist)
        n = len(hist)
        rec = {"inst": inst, "for_turn": r_next, "prompt_len": n,
               "pred_gpu": gpu, "pred_cpu": cpu, "t_issue": now(), "fired": False}
        if gpu >= 0.9 * n or (gpu + cpu) < 0.5 * n:
            rec["skip"] = "in_L1" if gpu >= 0.9 * n else "not_in_L2"
            pf_log.append(rec)
            return
        rec["fired"] = True
        pf_log.append(rec)

        mode = cfg["pf_mode"]
        if mode == "lowest":
            prio = int(now() * 1000) + 10**9
        elif mode == "bounded":
            prio = int((now() - cfg["jump_s"]) * 1000)
        else:
            prio = pf_prio
        rec["priority"] = prio

        def run():
            rec["t_send"] = now()
            try:
                r = post(url, model, hist, 1, prio)
            except Exception as e:
                rec["error"] = repr(e)
                return
            rec["t_recv"] = now()
            m = r.get("metrics") or {}
            rec["queue_ms"] = m.get("queue_time_ms")
            rec["ttft_ms"] = m.get("time_to_first_token_ms")
            rec["cached"] = ((r["usage"].get("prompt_tokens_details") or {})
                             .get("cached_tokens") or 0)
        pool.submit(run)

    def send(tid, prio, xargs=None):
        t = turns[tid]
        t.priority = prio
        if cfg.get("p_return_table"):
            tab = cfg["p_return_table"]
            pr = tab.get(str(t.r + 1), tab[max(tab, key=int)])
            xargs = {**(xargs or {}), "p_return": pr}
        t.pred_gpu, t.pred_cpu, _ = oracle.match(t.prompt)
        inflight.add(tid)

        def run():
            t.t_send = now()
            r = post(url, model, t.prompt, t.n_out, prio, xargs)
            t.t_recv = now()
            m = r.get("metrics") or {}
            t.queue_ms = m.get("queue_time_ms")
            t.ttft_ms = m.get("time_to_first_token_ms")
            t.gen_ms = m.get("generation_time_ms")
            u = r["usage"]
            t.cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            t.kvp = r.get("kv_transfer_params") or None
            evq.put(("done", tid))
        pool.submit(run)

    def plan(state: State):
        pending = list(state["pending"])
        if group in ("kvaware_prefetch", "kvaware_slo"):
            tnow = now()
            for inst, (due, r_next) in list(pf_due.items()):
                if tnow >= due:
                    del pf_due[inst]
                    prefetch(inst, r_next)
        if group == "baseline":
            for tid in pending:
                send(tid, 0)
            return {"pending": [], "events": state["events"] + 1}

        if group == "kvprotect":
            # everything goes to the engine at once; the engine protects a turn
            # once it has waited protect_s there (deadline promotion patch)
            for tid in pending:
                t = turns[tid]
                gpu, cpu, _ = oracle.match(t.prompt)
                if (gpu + cpu) >= 0.5 * len(t.prompt):
                    prio = -((gpu + cpu) // 16)
                elif cfg.get("cold_sjf"):
                    prio = max(1, (len(t.prompt) - gpu - cpu) // 256)     # shorter prefill first
                else:
                    prio = 0
                send(tid, prio, {"protect_s": cfg["protect_s"]})
            return {"pending": [], "events": state["events"] + 1}

        if group in ("kvaware_fair", "kvaware_slo"):
            jump_ms = int(cfg["jump_s"] * 1000)
            for tid in pending:
                t = turns[tid]
                gpu, cpu, _ = oracle.match(t.prompt)
                prio = int(t.t_arrive * 1000)
                if (gpu + cpu) >= 0.5 * len(t.prompt):
                    prio -= jump_ms            # bounded jump: at most J seconds
                send(tid, prio)
            return {"pending": [], "events": state["events"] + 1}

        st = engine.sample()
        used = max(sum(len(turns[x].prompt) for x in inflight), st["usage"] * capacity)
        budget = alpha * capacity
        scored = []
        for tid in pending:
            t = turns[tid]
            gpu, cpu, _ = oracle.match(t.prompt)
            scored.append((-(gpu + cpu), t.t_arrive, tid, gpu, cpu))
        scored.sort()
        keep = []
        for neg, _, tid, gpu, cpu in scored:
            t = turns[tid]
            if group == "kvaware_ft" and t.r == 0:
                # first turn: nothing cached to exploit; nobody may overtake it
                send(tid, -10**7 + int(t.t_arrive * 1000))
                used += len(t.prompt)
            elif (gpu + cpu) >= 0.5 * len(t.prompt):
                send(tid, -((gpu + cpu) // 16))          # warm: jump the queue
                used += len(t.prompt) - gpu
            elif used + len(t.prompt) <= budget or t.held >= cfg["max_hold"]:
                send(tid, 0)
                used += len(t.prompt)
            else:
                t.held += 1
                keep.append(tid)
        return {"pending": keep, "events": state["events"] + 1}

    def dispatch(state: State):
        return {}

    def collect(state: State):
        pending, recs, finished = list(state["pending"]), [], state["finished"]
        items = []
        try:
            items.append(evq.get(timeout=0.2))
        except queue.Empty:
            pass
        while True:
            try:
                items.append(evq.get_nowait())
            except queue.Empty:
                break
        for kind, tid in items:
            if kind == "arrive":
                pending.append(tid)
                pf_due.pop(turns[tid].inst, None)       # user came back first
            else:
                inflight.discard(tid)
                t = turns[tid]
                recs.append(t.row())
                finished += 1
                if group in ("kvaware_prefetch", "kvaware_slo") and t.r + 1 < len(convs[t.inst].prompts):
                    pf_due[t.inst] = (now() + pf_delay, t.r + 1)
                t.done_evt.set()
        return {"pending": pending, "finished": finished, "records": recs}

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


MC_BLOCK = 512
_mc_blocks: dict = {}


def mc_block_tokens(h):
    """Deterministic token block for a Mooncake hash id (plain vocab range, no specials)."""
    b = _mc_blocks.get(h)
    if b is None:
        rng = random.Random("mooncake:%d" % h)
        b = [rng.randrange(1000, 120000) for _ in range(MC_BLOCK)]
        _mc_blocks[h] = b
    return b


class MConv:
    """A replayed Mooncake session: one synthesized prompt per turn."""

    def __init__(self, k, se):
        self.id = "mc%d" % k
        self.prompts, self.out_lens = [], []
        for t in se["turns"]:
            toks = []
            for h in t["hash_ids"]:
                toks.extend(mc_block_tokens(h))
            self.prompts.append(toks[:t["input_length"]])
            self.out_lens.append(max(1, int(t["output_length"])))
        self.history = list(self.prompts)


def load_mooncake(wl):
    convs, arrivals, thinks, n_turns = [], [], [], []
    for k, se in enumerate(wl["sessions"]):
        c = MConv(k, se)
        convs.append(c)
        arrivals.append(float(se["start"]))
        thinks.append(list(se["thinks"]) + [0.0])
        n_turns.append(len(c.prompts))
    return convs, arrivals, thinks, n_turns


def load_workload(wl, spec, traj, seed):
    """sessions -> (convs, arrivals, thinks, n_turns); one unique instance per session."""
    sess = wl["sessions"]
    rng = random.Random("%d:workload" % seed)
    pool = sorted(spec, key=lambda i: i["id"])
    rng.shuffle(pool)
    if len(sess) > len(pool):
        raise SystemExit("workload has %d sessions but only %d unique instances" % (len(sess), len(pool)))
    convs, arrivals, thinks, n_turns = [], [], [], []
    for inst, se in zip(pool, sess):
        c = Conv(inst, traj[inst["id"]])
        n = min(se["n"], len(c.prompts))
        z = list(se["thinks"][:n - 1]) + [0.0]
        convs.append(c)
        arrivals.append(float(se["start"]))
        thinks.append(z)
        n_turns.append(n)
    return convs, arrivals, thinks, n_turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    choices=["baseline", "kvaware", "kvaware_fair", "kvaware_prefetch",
                             "kvaware_slo", "kvaware_ft", "kvprotect"])
    ap.add_argument("--pf-delay", type=float, default=2.5,
                    help="kvaware_prefetch: seconds after an answer before checking/prefetching")
    ap.add_argument("--pf-lowest", action="store_true",
                    help="alias for --pf-mode lowest")
    ap.add_argument("--pf-mode", choices=["fixed", "lowest", "bounded"], default=None,
                    help="prefetch priority; default lowest for kvaware_slo, fixed otherwise")
    ap.add_argument("--pf-prio", type=int, default=-1,
                    help="prefetch priority: below real warm turns, above cold (0)")
    ap.add_argument("--cold-sjf", action="store_true",
                    help="kvprotect: cold requests ordered shortest-prefill-first (bounded by protect_s)")
    ap.add_argument("--p-return-table", default=None,
                    help="JSON {depth: P(next turn)}; sends vllm_xargs.p_return with every request")
    ap.add_argument("--protect-s", type=float, default=60.0,
                    help="kvprotect: engine-side protection after this many seconds of waiting")
    ap.add_argument("--jump-s", type=float, default=60.0,
                    help="kvaware_fair: a warm turn may overtake cold turns that "
                         "arrived at most this many seconds before it")
    ap.add_argument("--instances", default="data/instances_320.json")
    ap.add_argument("--traj", default="trajectories/trajectories_320.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workload", default=None,
                    help="trace-calibrated sessions (build_workload.py); overrides --n/arrival/think")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--arrival-mean", type=float, default=50.0)
    ap.add_argument("--arrival-std", type=float, default=20.0)
    ap.add_argument("--arrival-window", type=float, default=100.0)
    ap.add_argument("--arrival-dist", choices=["normal", "poisson"], default="normal",
                    help="poisson: Poisson process conditioned on n arrivals in [0, window]")
    ap.add_argument("--think-mean", type=float, default=5.0)
    ap.add_argument("--think-std", type=float, default=1.0)
    ap.add_argument("--think-min", type=float, default=0.5)
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--events", default="tcp://127.0.0.1:5557")
    ap.add_argument("--capacity", type=int, default=518080)
    ap.add_argument("--gpu", type=int, default=0,
                    help="card index for nvidia-smi sampling")
    ap.add_argument("--sample-s", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-hold", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())["instances"]
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    rng = random.Random(args.seed)
    picked = rng.sample(spec, min(args.n, len(spec)))
    convs = [Conv(i, traj[i["id"]]) for i in picked]

    # pre-sample the users: first-turn arrival and every think time
    arr = random.Random("%d:arrival" % args.seed)
    if args.arrival_dist == "poisson":
        # n arrivals of a Poisson process given that all n fall in [0, window]:
        # the arrival times are n sorted i.i.d. uniforms (exponential gaps)
        ts = sorted(arr.uniform(0.0, args.arrival_window) for _ in convs)
        order = list(range(len(convs)))
        arr.shuffle(order)
        arrivals = [0.0] * len(convs)
        for k, i in enumerate(order):
            arrivals[i] = ts[k]
    else:
        arrivals = [min(args.arrival_window, max(0.0, arr.gauss(args.arrival_mean, args.arrival_std)))
                    for _ in convs]
    thinks = [[max(args.think_min,
                   random.Random("%d:think:%s:%d" % (args.seed, c.id, r))
                   .gauss(args.think_mean, args.think_std))
               for r in range(len(c.prompts))] for c in convs]
    total = sum(len(c.prompts) for c in convs)
    n_turns = [len(c.prompts) for c in convs]
    workload_meta = None
    if args.workload:
        wl = json.loads(Path(args.workload).read_text())
        workload_meta = wl["meta"]
        if wl["meta"].get("kind") == "mooncake":
            convs, arrivals, thinks, n_turns = load_mooncake(wl)
        else:
            convs, arrivals, thinks, n_turns = load_workload(wl, spec, traj, args.seed)
        total = sum(n_turns)

    oracle = CacheOracle(args.events).start()
    engine = EngineState(args.base)
    time.sleep(1.0)
    root = args.base.rstrip("/").rsplit("/v1", 1)[0]
    reset_all_caches(root, args.base, args.model, convs[0].prompts[0])
    time.sleep(2.0)

    sampler = Sampler(args.base, args.gpu, lambda: now() - t_zero[0], args.sample_s)
    evq: queue.Queue = queue.Queue()
    turns: dict = {}
    pf_log: list = []
    graph = build({"convs": convs, "pf_log": pf_log,
                   "pf_delay": args.pf_delay, "pf_prio": args.pf_prio,
                   "pf_mode": (args.pf_mode or ("lowest" if (args.pf_lowest or
                               args.group == "kvaware_slo") else "fixed")),"oracle": oracle, "engine": engine, "group": args.group,
                   "url": args.base.rstrip("/") + "/completions", "model": args.model,
                   "capacity": args.capacity, "alpha": args.alpha, "total": total,
                   "events": evq, "turns": turns, "max_hold": args.max_hold,
                   "jump_s": args.jump_s, "protect_s": args.protect_s,
                   "p_return_table": json.loads(args.p_return_table) if args.p_return_table else None,
                   "cold_sjf": args.cold_sjf})

    t_zero = [None]

    def user(i):
        c = convs[i]
        time.sleep(max(0.0, t_zero[0] + arrivals[i] - now()))
        for r in range(n_turns[i]):
            t = Turn(i, r, c.prompts[r], c.out_lens[r], thinks[i][r])
            tid = id(t)
            turns[tid] = t
            t.t_arrive = now()
            evq.put(("arrive", tid))
            t.done_evt.wait()
            if r + 1 < n_turns[i]:
                time.sleep(thinks[i][r])

    lens = [len(c.prompts[0]) for c in convs]
    print("group=%s instances=%d turns=%d" % (args.group, len(convs), total))
    print("turn-1 arrival ~ N(%.0f, %.0f) clipped to [0, %.0f]s; think ~ N(%.1f, %.1f) >= %.1fs"
          % (args.arrival_mean, args.arrival_std, args.arrival_window,
             args.think_mean, args.think_std, args.think_min))
    print("turn-1 prompt tokens: min %d  mean %d  max %d  (total %d vs pool %d)"
          % (min(lens), sum(lens) // len(lens), max(lens), sum(lens), args.capacity))

    t_zero[0] = now()
    threads = [threading.Thread(target=user, args=(i,), daemon=True)
               for i in range(len(convs))]
    for th in threads:
        th.start()
    sampler.start()
    final = graph.invoke({"pending": [], "finished": 0, "events": 0, "records": []},
                         {"recursion_limit": 10_000_000})
    oracle.stop()

    sampler.stop()
    R = final["records"]
    for d in R:                     # rebase all times to the first arrival
        for k in ("t_arrive", "t_send", "t_recv", "prefill_start",
                  "decode_start", "decode_end"):
            d[k] -= t_zero[0]
    for p in pf_log:
        for k in ("t_issue", "t_send", "t_recv"):
            if p.get(k) is not None:
                p[k] -= t_zero[0]
    fired = [p for p in pf_log if p["fired"]]
    if args.group in ("kvaware_prefetch", "kvaware_slo"):
        print("prefetch: checked %d, fired %d, skipped in_L1 %d, skipped not_in_L2 %d"
              % (len(pf_log), len(fired),
                 sum(1 for p in pf_log if p.get("skip") == "in_L1"),
                 sum(1 for p in pf_log if p.get("skip") == "not_in_L2")))
    later = [d for d in R if d["r"] > 0]
    hit2 = 100 * sum(d["cached"] for d in later) / max(1, sum(d["prompt_len"] for d in later))
    wait = sorted(d["prefill_start"] - d["t_arrive"] for d in R)
    resp = sorted(d["decode_end"] - d["t_arrive"] for d in R)
    over = sorted(d["overhead_s"] for d in R)
    missing = sum(1 for d in R if d["ttft_ms"] is None)
    q = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))]
    print("\nmakespan %.1fs over %d turns" % (max(d["decode_end"] for d in R), len(R)))
    print("wait      p50 %.2fs  p90 %.2fs  p99 %.2fs" % (q(wait, .5), q(wait, .9), q(wait, .99)))
    print("response  p50 %.2fs  p90 %.2fs  p99 %.2fs" % (q(resp, .5), q(resp, .9), q(resp, .99)))
    print("prefix hit on turns 2+: %.1f%%" % hit2)
    print("unaccounted client overhead p50 %.3fs p99 %.3fs  (turns missing engine metrics: %d)"
          % (q(over, .5), q(over, .99), missing))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "group": args.group, "n": args.n, "seed": args.seed,
        "arrival": [args.arrival_mean, args.arrival_std, args.arrival_window],
        "arrival_dist": args.arrival_dist,
        "workload": workload_meta, "workload_file": args.workload,
        "think": [args.think_mean, args.think_std, args.think_min],
        "capacity": args.capacity, "jump_s": args.jump_s, "protect_s": args.protect_s,
        "p_return_table": args.p_return_table,
        "cold_sjf": args.cold_sjf,
        "pf_delay": args.pf_delay,
        "pf_mode": (args.pf_mode or ("lowest" if (args.pf_lowest or
                    args.group == "kvaware_slo") else "fixed")),
        "prefetches": pf_log, "samples": sampler.samples, "records": R,
    }))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
