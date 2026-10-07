"""Closed-loop interactive replay: users, think time, and a harness that queues.

The one thing this defines is when a user's next turn arrives:

    t_arrive[u, r+1] = t_done[u, r] + Z[u, r]

Z is the user's think time after reading answer r. Everything else follows.
There is no fixed timestamp table: a faster system gets its next turns sooner,
exactly like real users. U users are online at once; a user finishes one
conversation and starts the next, so first-turn arrivals need no separate
definition either. Z is pre-sampled with seed(u, r) so every arm sees the
same think times and differs only in scheduling.

Because arrivals depend on completions, this is a closed system with positive
feedback: misses cause recompute, recompute evicts more, more misses. Good
scheduling keeps it stable; bad scheduling lets the queue explode. So besides
makespan we report per-turn response time (arrival -> completion), which is
what a user actually feels.

Arms:
    baseline  FIFO on arrival, priority 0, dispatch everything immediately.
    kvaware   a turn whose prefix the engine still holds jumps the queue (sent
              with high priority so the engine admits it first and evicts it
              last); a cold turn is held back while the pool is committed, and
              released FIFO as room frees; an optional prefetch pulls a
              thinking user's KV back from the CPU tier before they return.

The harness is still one LangGraph over the whole run:
    START -> plan -> dispatch -> collect -> (turns left ? plan : END)
with collect blocking on the next event (arrival or completion).
"""

import argparse
import json
import math
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


def now():
    return time.perf_counter() - T0


def post(url, model, prompt, n_out, priority=0):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        r = json.loads(resp.read())
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return u["prompt_tokens"], cached


class Conv:
    def __init__(self, inst, traj):
        self.id = inst["id"]
        self.prompts, self.prefix_after = [], []
        ids = list(inst["head_ids"])
        for t, turn in enumerate(traj["turns"]):
            self.prompts.append(ids)
            after = ids + turn["output_ids"]         # what the engine holds
            self.prefix_after.append(after)          # once answer t is done
            if t < len(inst["turn_suffix_ids"]):
                ids = after + inst["turn_suffix_ids"][t]
        self.out_lens = [len(t["output_ids"]) for t in traj["turns"]]


class Turn:
    __slots__ = ("cid", "r", "prompt", "n_out", "t_arrive", "t_dispatch",
                 "t_done", "cached", "priority", "held", "gpu", "cpu", "done_evt")

    def __init__(self, cid, r, prompt, n_out):
        self.cid, self.r, self.prompt, self.n_out = cid, r, prompt, n_out
        self.t_arrive = self.t_dispatch = self.t_done = None
        self.cached = 0
        self.priority = 0
        self.held = 0
        self.gpu = self.cpu = 0
        self.done_evt = threading.Event()

    def row(self):
        return {"conv": self.cid, "turn": self.r, "prompt_len": len(self.prompt),
                "cached_tokens": self.cached, "priority": self.priority,
                "t_arrive": self.t_arrive, "t_dispatch": self.t_dispatch,
                "t_done": self.t_done,
                "queue_s": self.t_dispatch - self.t_arrive,
                "response_s": self.t_done - self.t_arrive,
                "held_rounds": self.held, "pred_gpu": self.gpu, "pred_cpu": self.cpu}


class State(TypedDict):
    pending: list           # arrived turn ids not yet dispatched
    finished: int
    events: int
    records: Annotated[list, lambda a, b: a + b]
    samples: Annotated[list, lambda a, b: a + b]


def build(cfg):
    convs, oracle, engine = cfg["convs"], cfg["oracle"], cfg["engine"]
    url, model, arm = cfg["url"], cfg["model"], cfg["arm"]
    capacity, alpha = cfg["capacity"], cfg["alpha"]
    total = cfg["total_turns"]
    evq: queue.Queue = cfg["events"]
    turns: dict = cfg["turns"]          # id -> Turn
    thinking: dict = cfg["thinking"]    # cid -> (t_done, prefix_after)
    pool = ThreadPoolExecutor(max_workers=256)
    box = {"inflight": {}, "prefetched": set(), "last_prefetch": 0.0}

    def committed():
        return sum(len(turns[t].prompt) for t in box["inflight"])

    def send(tid, prio):
        t = turns[tid]
        t.priority = prio
        t.t_dispatch = now()
        box["inflight"][tid] = True

        def run():
            plen, cached = post(url, model, t.prompt, t.n_out, prio)
            t.cached = cached
            t.t_done = now()
            evq.put(("done", tid))
        pool.submit(run)

    def prefetch(cid, prefix):
        # a 1-token request on the prefix makes the connector pull it GPU-side
        def run():
            try:
                post(url, model, prefix, 1, 1000)   # lowest priority
            except Exception:
                pass
        box["prefetched"].add(cid)
        pool.submit(run)

    def plan(state: State):
        pending = list(state["pending"])
        sample = {"t": now(), "pending": len(pending),
                  "inflight": len(box["inflight"])}
        if arm == "baseline":
            for tid in pending:
                send(tid, 0)
            return {"pending": [], "events": state["events"] + 1,
                    "samples": [sample]}

        st = engine.sample()
        usage_tokens = max(committed(), st["usage"] * capacity)
        budget = alpha * capacity
        sample.update(kv_usage=round(st["usage"], 3),
                      preemptions=st["preemptions"])

        scored = []
        for tid in pending:
            t = turns[tid]
            t.gpu, t.cpu, _ = oracle.match(t.prompt)
            scored.append(t)
        # warm turns first (their prefix is perishable), then FIFO among cold
        scored.sort(key=lambda t: (-(t.gpu + t.cpu), t.t_arrive))

        keep = []
        for t in scored:
            warm = (t.gpu + t.cpu) >= 0.5 * len(t.prompt)
            if warm:
                # jump the queue; the engine admits it first and preempts it last
                send(id(t), -((t.gpu + t.cpu) // 16))
                usage_tokens += len(t.prompt) - t.gpu
            elif (usage_tokens + len(t.prompt) <= budget
                  or t.held >= cfg["max_hold"]):
                send(id(t), 0)
                usage_tokens += len(t.prompt)
            else:
                t.held += 1
                keep.append(id(t))

        if cfg["prefetch"] and st["usage"] < cfg["prefetch_below"] \
                and now() - box["last_prefetch"] > 0.5:
            # pull back the thinking user most likely to return next whose KV
            # is only in the CPU tier
            best = None
            for cid, (t_done, prefix) in list(thinking.items()):
                if cid in box["prefetched"]:
                    continue
                gpu, cpu, _ = oracle.match(prefix)
                if cpu > 0 and gpu < 0.5 * len(prefix):
                    if best is None or t_done < best[0]:
                        best = (t_done, cid, prefix)
            if best:
                prefetch(best[1], best[2])
                box["last_prefetch"] = now()
                sample["prefetch"] = best[1]

        sample["held"] = len(keep)
        return {"pending": keep, "events": state["events"] + 1,
                "samples": [sample]}

    def dispatch(state: State):
        return {}

    def collect(state: State):
        pending = list(state["pending"])
        recs, finished = [], state["finished"]
        try:
            kind, tid = evq.get(timeout=0.25)
            items = [(kind, tid)]
        except queue.Empty:
            items = []
        while True:                      # drain whatever else is ready
            try:
                items.append(evq.get_nowait())
            except queue.Empty:
                break
        for kind, tid in items:
            if kind == "arrive":
                pending.append(tid)
                thinking.pop(turns[tid].cid, None)
                box["prefetched"].discard(turns[tid].cid)
            else:
                t = turns[tid]
                box["inflight"].pop(tid, None)
                recs.append(t.row())
                finished += 1
                c = convs[t.cid]
                if t.r + 1 < len(c.prompts):
                    thinking[t.cid] = (t.t_done, c.prefix_after[t.r])
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


def sample_think(seed, median, sigma):
    rng = random.Random(seed)
    return median * math.exp(sigma * rng.gauss(0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "kvaware"])
    ap.add_argument("--instances", default="data/instances_100.json")
    ap.add_argument("--traj", default="trajectories/trajectories_100.json")
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--events", default="tcp://127.0.0.1:5557")
    ap.add_argument("--users", type=int, default=40,
                    help="U: users online at once; each runs conversations back to back")
    ap.add_argument("--think-median", type=float, default=15.0,
                    help="median of the log-normal think time Z, seconds")
    ap.add_argument("--think-sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--capacity", type=int, default=518080)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-hold", type=int, default=40,
                    help="planning rounds a cold turn may be held before release")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--prefetch-below", type=float, default=0.75)
    ap.add_argument("--ramp-s", type=float, default=0.0,
                    help="spread user start times uniformly over this many seconds")
    ap.add_argument("--report-after", type=float, default=0.0,
                    help="headline metrics only over turns arriving after this time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    convs = {i["id"]: Conv(i, traj[i["id"]]) for i in spec["instances"]}
    order = [i["id"] for i in spec["instances"]]
    total_turns = sum(len(c.prompts) for c in convs.values())

    # pre-sample every think time so all arms see identical users
    think = {cid: [sample_think("%d:%s:%d" % (args.seed, cid, r), args.think_median, args.think_sigma)
                   for r in range(len(convs[cid].prompts))]
             for cid in order}

    oracle = CacheOracle(args.events).start()
    engine = EngineState(args.base)
    time.sleep(1.0)
    root = args.base.rstrip("/").rsplit("/v1", 1)[0]
    urllib.request.urlopen(urllib.request.Request(
        root + "/reset_prefix_cache", data=b"",
        headers={"Content-Type": "application/json"}), timeout=60).read()
    time.sleep(2.0)

    evq: queue.Queue = queue.Queue()
    turns: dict = {}
    thinking: dict = {}
    graph = build({
        "convs": convs, "oracle": oracle, "engine": engine, "arm": args.arm,
        "url": args.base.rstrip("/") + "/completions", "model": args.model,
        "capacity": args.capacity, "alpha": args.alpha,
        "total_turns": total_turns, "events": evq, "turns": turns,
        "thinking": thinking, "max_hold": args.max_hold,
        "prefetch": args.prefetch, "prefetch_below": args.prefetch_below,
    })

    # user u handles conversations u, u+U, u+2U, ... back to back
    def user(u):
        if args.ramp_s > 0:
            time.sleep(u * args.ramp_s / args.users)   # staggered login
        for cid in order[u::args.users]:
            c = convs[cid]
            t_prev_done = None
            for r in range(len(c.prompts)):
                if t_prev_done is not None:
                    wake = t_prev_done + think[cid][r - 1]
                    time.sleep(max(0.0, wake - now()))
                t = Turn(cid, r, c.prompts[r], c.out_lens[r])
                turns[id(t)] = t
                t.t_arrive = now()
                evq.put(("arrive", id(t)))
                t.done_evt.wait()
                t_prev_done = t.t_done

    print("arm=%s users=%d conversations=%d turns=%d think~LogNormal(median=%.0fs, sigma=%.1f)"
          % (args.arm, args.users, len(convs), total_turns,
             args.think_median, args.think_sigma))
    print("pool %d tokens ~ %.0f conversations of mean length %d; %d users online"
          % (args.capacity,
             args.capacity / (sum(len(c.prompts[0]) for c in convs.values()) / len(convs)),
             sum(len(c.prompts[0]) for c in convs.values()) // len(convs), args.users))

    threads = [threading.Thread(target=user, args=(u,), daemon=True)
               for u in range(args.users)]
    t_start = now()
    for th in threads:
        th.start()
    final = graph.invoke({"pending": [], "finished": 0, "events": 0,
                          "records": [], "samples": []},
                         {"recursion_limit": 10_000_000})
    wall = now() - t_start
    for th in threads:
        th.join(timeout=5)
    oracle.stop()

    R_all = final["records"]
    R = [r for r in R_all if r["t_arrive"] >= args.report_after] or R_all
    if args.report_after > 0:
        print("\n[steady state: %d of %d turns arrived after t=%.0fs]"
              % (len(R), len(R_all), args.report_after))
    resp = sorted(r["response_s"] for r in R)
    q = sorted(r["queue_s"] for r in R)
    later = [r for r in R if r["turn"] > 0]
    lhit = sum(r["cached_tokens"] for r in later)
    ltot = max(1, sum(r["prompt_len"] for r in later))
    tot = sum(r["prompt_len"] for r in R)
    hit = sum(r["cached_tokens"] for r in R)
    pend = [s["pending"] for s in final["samples"]]
    infl = [s["inflight"] for s in final["samples"]]

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print("\nmakespan %.1fs  (%d turns, %d scheduling events)" % (wall, len(R), final["events"]))
    print("response time  p50 %.2fs  p90 %.2fs  p99 %.2fs" % (pct(resp, .5), pct(resp, .9), pct(resp, .99)))
    print("queue wait     p50 %.2fs  p90 %.2fs  p99 %.2fs" % (pct(q, .5), pct(q, .9), pct(q, .99)))
    print("prefix hit     %.1f%% overall, %.1f%% on turns 2+" % (100 * hit / tot, 100 * lhit / ltot))
    print("recomputed     %.2fM of %.2fM prompt tokens" % ((tot - hit) / 1e6, tot / 1e6))
    print("queue length   med %d  max %d   |  in flight med %d  max %d"
          % (sorted(pend)[len(pend) // 2], max(pend), sorted(infl)[len(infl) // 2], max(infl)))
    held = sum(1 for r in R if r["held_rounds"] > 0)
    print("held-back cold turns: %d/%d" % (held, len(R)))
    print("oracle %s" % oracle.snapshot())

    out = args.out or "runs/closed-%s.json" % args.arm
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "arm": args.arm, "users": args.users, "think_median": args.think_median,
        "think_sigma": args.think_sigma, "seed": args.seed, "capacity": args.capacity,
        "makespan_s": wall, "ramp_s": args.ramp_s,
        "report_after": args.report_after,
        "records": R_all, "samples": final["samples"],
    }))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
