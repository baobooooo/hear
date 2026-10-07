"""Trace-calibrated closed-loop workload from BurstGPT sessions.

Timing comes from the production trace (BurstGPT_3, conversation sessions);
content comes from SCBench (long-context multi-turn instances), because that is
the setting where KV reuse matters. Each replayed session u keeps
    * its own turn count n_u (capped by the SCBench instance's turns),
    * its own think-time sequence z_{u,1..n-1} (the real gaps, capped),
and the harness issues turn r+1 at f_{u,r} + z_{u,r}, where f is the tested
system's actual completion time (closed loop).

Session start times, three modes at the same mean rate R (sessions / min):
    trace    real start offsets from the busiest trace hours, superposed until
             R*H/60 sessions fill the horizon H (keeps intra-hour burstiness)
    poisson  exponential gaps (matched-rate control)
    gamma    gamma renewal with the given CV (burstier / smoother control)

usage: build_workload.py --rate 6 --horizon 900 --mode trace --seed 2026 --out workloads/trace-r6.json
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/burstgpt/sessions.json")
    ap.add_argument("--rate", type=float, required=True, help="sessions per minute")
    ap.add_argument("--horizon", type=float, default=900.0, help="seconds of session arrivals")
    ap.add_argument("--mode", choices=["trace", "poisson", "gamma"], default="trace")
    ap.add_argument("--cv", type=float, default=2.0, help="gamma renewal CV")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=6, help="SCBench instances have <= 6 turns")
    ap.add_argument("--think-cap", type=float, default=300.0, help="clip think times (s)")
    ap.add_argument("--from", dest="base", default=None,
                    help="matched control: reuse this workload's sessions (turns, think times), redraw starts only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random("%d:%s:%g" % (a.seed, a.mode, a.rate))
    S = [s for s in json.load(open(a.sessions))["sessions"] if s["n"] >= a.min_turns]
    M = int(round(a.rate * a.horizon / 60.0))

    def shape(s):
        n = min(s["n"], a.max_turns)
        return {"n": n, "thinks": [min(a.think_cap, z) for z in s["thinks"][:n - 1]],
                "trace_n": s["n"], "trace_start": s["start"]}

    if a.base:
        base = json.load(open(a.base))
        sess = [dict(s) for s in base["sessions"]]
        M = len(sess)
        assert a.mode != "trace", "--from is for the poisson/gamma controls"
        mean_gap = a.horizon / M
        if a.mode == "poisson":
            gaps = [rng.expovariate(1.0 / mean_gap) for _ in range(M)]
        else:
            k = 1.0 / (a.cv ** 2)
            gaps = [rng.gammavariate(k, mean_gap / k) for _ in range(M)]
        t, starts = 0.0, []
        for g in gaps:
            t += g
            starts.append(t)
        scale = a.horizon / starts[-1]
        starts = [t * scale for t in starts]
        rng.shuffle(sess)                       # which session gets which slot is arbitrary
    elif a.mode == "trace":
        blocks = defaultdict(list)
        for s in S:
            blocks[(int(s["start"] // 86400), int(s["start"] % 86400 // 3600))].append(s)
        # busiest hours first, in seeded order among the top 40
        top = sorted(blocks.values(), key=len, reverse=True)[:40]
        rng.shuffle(top)
        out = []
        for blk in top:
            for s in blk:
                off = (s["start"] % 3600) * (a.horizon / 3600.0)      # keep the shape of the hour
                out.append((off, shape(s)))
            if len(out) >= M:
                break
        rng.shuffle(out)
        out = sorted(out[:M], key=lambda x: x[0])
        starts = [o for o, _ in out]
        sess = [s for _, s in out]
    else:
        sess = [shape(s) for s in rng.sample(S, M)]
        mean_gap = 60.0 / a.rate
        if a.mode == "poisson":
            gaps = [rng.expovariate(1.0 / mean_gap) for _ in range(M)]
        else:
            k = 1.0 / (a.cv ** 2)
            gaps = [rng.gammavariate(k, mean_gap / k) for _ in range(M)]
        t, starts = 0.0, []
        for g in gaps:
            t += g
            starts.append(t)
        scale = a.horizon / starts[-1]                      # exactly R over the horizon
        starts = [t * scale for t in starts]

    for st, s in zip(starts, sess):
        s["start"] = round(st, 3)
    sess.sort(key=lambda s: s["start"])
    st = [s["start"] for s in sess]
    gaps = [b - a_ for a_, b in zip(st, st[1:])]
    mg = sum(gaps) / len(gaps)
    gap_cv = (sum((g - mg) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mg
    turns = sum(s["n"] for s in sess)
    thinks = [z for s in sess for z in s["thinks"]]
    thinks.sort()
    q = lambda p: thinks[min(len(thinks) - 1, int(len(thinks) * p))] if thinks else 0
    meta = {"mode": a.mode, "rate_per_min": a.rate, "horizon_s": a.horizon, "cv": a.cv if a.mode == "gamma" else None,
            "seed": a.seed, "sessions": M, "turns": turns, "think_cap_s": a.think_cap,
            "think_p50": q(.5), "think_p90": q(.9), "turns_per_session": turns / M,
            "gap_cv": round(gap_cv, 3), "matched_from": a.base, "source": a.sessions}
    json.dump({"meta": meta, "sessions": sess}, open(a.out, "w"))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
