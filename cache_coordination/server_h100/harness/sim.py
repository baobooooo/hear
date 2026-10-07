"""Offline simulator for the closed-loop interactive scenario.

Purpose: find the number of online users U at which, on the real single-card
KV pool, a returning user's prefix is sometimes still resident and sometimes
not -- and see whether FIFO dispatch tips into the unstable state while
cache-aware dispatch does not. Cheap to sweep; no GPU.

Model (all rates measured on the actual engine):
  * GPU: prefill shared at PREFILL tok/s across in-flight requests; decode
    per request min(DEC_SOLO, DEC_AGG / n_decoding) tok/s.
  * KV pool: CAPACITY tokens. An admitted request needs its whole prompt
    resident. LRU over conversations that are not in flight; if nothing can be
    evicted the request waits in the engine queue (this is what vLLM does).
  * Engine queue order: (priority, arrival).
  * Users: U online, each runs conversations back to back; turn r+1 arrives at
    t_done(r) + Z, Z ~ LogNormal(median m, sigma), seeded per (u, conv, r).
  * Harness arms: baseline = submit on arrival, priority 0.
                  kvaware  = warm turn -> submit with high priority; cold turn
                             held while resident usage > alpha*CAPACITY,
                             released FIFO as room frees (starvation cap).
"""

import argparse
import json
import math
import random
import statistics
from collections import OrderedDict
from pathlib import Path

PREFILL = 17_600.0      # tok/s, cold prefill throughput (measured)
DEC_SOLO = 120.0        # tok/s per request at low batch (measured 134 @20k)
DEC_AGG = 350.0         # tok/s aggregate (measured 342 @ bs 8)
CAPACITY = 518_080
DT = 0.05


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def load_convs(path, n):
    spec = json.loads(Path(path).read_text())["instances"][:n]
    out = []
    for inst in spec:
        turns = len(inst["turn_suffix_ids"]) + 1
        lens = [inst["turn1_len"]]
        for s in inst["turn_suffix_ids"]:
            lens.append(lens[-1] + 60 + len(s))       # ~60 answer tokens
        out.append({"id": inst["id"], "lens": lens, "out": [60] * turns})
    return out


def simulate(convs, U, m, sigma, arm, alpha=0.85, seed=7, max_hold_s=30.0,
             capacity=CAPACITY, verbose=False):
    rng_z = lambda u, c, r: m * math.exp(sigma * random.Random(
        "%d:%d:%s:%d" % (seed, u, c, r)).gauss(0, 1))

    # user state
    assign = {u: [i for i in range(len(convs)) if i % U == u] for u in range(U)}
    users = {u: {"queue": list(assign[u]), "conv": None, "r": 0,
                 "wake": 0.0, "waiting": False} for u in range(U)}
    for u in users.values():
        if u["queue"]:
            u["conv"] = u["queue"].pop(0)

    cache = OrderedDict()           # conv -> resident tokens (LRU order)
    inflight = {}                   # req id -> dict
    harness_q = []                  # cold turns held by harness
    engine_q = []                   # (priority, arrival, req)
    done_rows = []
    t = 0.0
    nid = 0
    total_turns = sum(len(c["lens"]) for c in convs)
    samples = []

    def resident_total():
        return sum(cache.values())

    def try_admit():
        # engine: admit from its queue in (priority, arrival) order while the
        # request's blocks fit after evicting idle LRU conversations
        engine_q.sort(key=lambda x: (x[0], x[1]))
        admitted = []
        for item in list(engine_q):
            prio, arr, req = item
            need = req["len"]
            hit = cache.get(req["conv"], 0) if req["conv"] in cache else 0
            # evict idle LRU until it fits
            while resident_total() - hit + need > capacity:
                victim = next((c for c in cache if c not in busy), None)
                if victim is None:
                    break
                cache.pop(victim)
            if resident_total() - hit + need > capacity:
                continue
            if req["conv"] in cache:
                cache.move_to_end(req["conv"])
            cache[req["conv"]] = need
            busy.add(req["conv"])
            req["hit"] = min(hit, need)
            req["pre_left"] = max(0, need - req["hit"])
            req["dec_left"] = req["out"]
            req["t_dispatch_engine"] = t
            inflight[req["id"]] = req
            engine_q.remove(item)
            admitted.append(req)
        return admitted

    busy = set()
    while len(done_rows) < total_turns and t < 20_000:
        # ---- arrivals
        for u, st in users.items():
            if st["conv"] is None or st["waiting"] or t < st["wake"]:
                continue
            c = convs[st["conv"]]
            req = {"id": nid, "u": u, "conv": st["conv"], "r": st["r"],
                   "len": c["lens"][st["r"]], "out": c["out"][st["r"]],
                   "t_arrive": t, "held": 0.0}
            nid += 1
            st["waiting"] = True
            warm = st["conv"] in cache and cache[st["conv"]] >= 0.5 * req["len"]
            req["warm"] = warm
            if arm == "baseline":
                engine_q.append((0, t, req))
            else:
                if warm:
                    engine_q.append((-cache[st["conv"]] // 16, t, req))
                else:
                    harness_q.append(req)

        # ---- harness releases cold turns when the pool has room
        if arm == "kvaware":
            budget = alpha * capacity
            used = resident_total()
            still = []
            for req in harness_q:
                if used + req["len"] <= budget or (t - req["t_arrive"]) > max_hold_s:
                    engine_q.append((0, t, req))
                    used += req["len"]
                else:
                    req["held"] += DT
                    still.append(req)
            harness_q = still

        try_admit()

        # ---- serve
        pre = [r for r in inflight.values() if r["pre_left"] > 0]
        dec = [r for r in inflight.values() if r["pre_left"] <= 0 and r["dec_left"] > 0]
        if pre:
            share = PREFILL * DT / len(pre)
            for r in pre:
                r["pre_left"] -= share
        if dec:
            rate = min(DEC_SOLO, DEC_AGG / len(dec))
            for r in dec:
                r["dec_left"] -= rate * DT

        for rid, r in list(inflight.items()):
            if r["pre_left"] <= 0 and r["dec_left"] <= 0:
                inflight.pop(rid)
                busy.discard(r["conv"])
                st = users[r["u"]]
                c = convs[r["conv"]]
                cache[r["conv"]] = c["lens"][r["r"]] + r["out"]
                cache.move_to_end(r["conv"])
                done_rows.append({"r": r["r"], "len": r["len"], "hit": r["hit"],
                                  "resp": t - r["t_arrive"], "held": r["held"]})
                st["waiting"] = False
                if r["r"] + 1 < len(c["lens"]):
                    st["r"] += 1
                    st["wake"] = t + rng_z(r["u"], r["conv"], r["r"])
                else:
                    st["r"] = 0
                    st["conv"] = st["queue"].pop(0) if st["queue"] else None
                    st["wake"] = t

        if int(t / DT) % 20 == 0:
            samples.append((t, len(engine_q) + len(harness_q), len(inflight)))
        t += DT

    later = [d for d in done_rows if d["r"] > 0]
    hit = sum(d["hit"] for d in later)
    tot = sum(d["len"] for d in later)
    resp = [d["resp"] for d in done_rows]
    qlen = [s[1] for s in samples]
    tail = qlen[-max(1, len(qlen) // 5):]
    return {
        "U": U, "arm": arm, "makespan": t,
        "hit2": 100 * hit / max(1, tot),
        "p50": pct(resp, .5), "p99": pct(resp, .99),
        "qmax": max(qlen) if qlen else 0,
        "qtail": statistics.mean(tail) if tail else 0,
        "finished": len(done_rows), "total": total_turns,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/instances_320.json")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--users", default="40,60,80,100,120,150")
    ap.add_argument("--think-median", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()
    convs = load_convs(args.instances, args.n)
    mean_len = statistics.mean(c["lens"][0] for c in convs)
    print("conversations=%d  mean turn-1 len=%d  pool holds ~%.1f  think median=%.0fs sigma=%.1f"
          % (len(convs), mean_len, CAPACITY / mean_len, args.think_median, args.sigma))
    print("%5s %-9s %9s %8s %7s %7s %6s %7s %s" % (
        "U", "arm", "makespan", "hit2+%", "p50", "p99", "qmax", "qtail", "done"))
    for U in [int(x) for x in args.users.split(",")]:
        for arm in ("baseline", "kvaware"):
            r = simulate(convs, U, args.think_median, args.sigma, arm)
            print("%5d %-9s %8.0fs %7.1f%% %6.1fs %6.1fs %6d %7.1f %d/%d" % (
                U, arm, r["makespan"], r["hit2"], r["p50"], r["p99"],
                r["qmax"], r["qtail"], r["finished"], r["total"]))


if __name__ == "__main__":
    main()
