"""Compare closed-loop runs: steady-state headline numbers side by side, plus a
time-bucketed view so we can see whether an arm ever tips into the unstable
state (hit rate collapses, response time climbs) and when it recovers.

usage: analyze_closed.py runs/closed/s320-u30-baseline.json runs/closed/s320-u30-kvaware.json
"""

import json
import sys
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def headline(d):
    R_all = d["records"]
    after = d.get("report_after", 0.0)
    R = [r for r in R_all if r["t_arrive"] >= after] or R_all
    later = [r for r in R if r["turn"] > 0]
    tot = sum(r["prompt_len"] for r in R)
    hit = sum(r["cached_tokens"] for r in R)
    ltot = max(1, sum(r["prompt_len"] for r in later))
    lhit = sum(r["cached_tokens"] for r in later)
    resp = [r["response_s"] for r in R]
    q = [r["queue_s"] for r in R]
    infl = [s["inflight"] for s in d["samples"] if s["t"] >= after] or [0]
    return {
        "arm": d["arm"], "users": d["users"], "turns": len(R), "all_turns": len(R_all),
        "makespan": d["makespan_s"],
        "hit2": 100 * lhit / ltot, "hit_all": 100 * hit / tot,
        "recompute_M": (tot - hit) / 1e6,
        "p50": pct(resp, .5), "p90": pct(resp, .9), "p99": pct(resp, .99),
        "q50": pct(q, .5), "q99": pct(q, .99),
        "infl_med": pct(infl, .5), "infl_max": max(infl),
        "held": sum(1 for r in R if r["held_rounds"] > 0),
    }


def buckets(d, w):
    R = d["records"]
    T = max(r["t_done"] for r in R)
    out = []
    for b in range(0, int(T) + w, w):
        rs = [r for r in R if b <= r["t_done"] < b + w and r["turn"] > 0]
        if not rs:
            continue
        tok = sum(r["prompt_len"] for r in rs)
        hit = sum(r["cached_tokens"] for r in rs)
        out.append((b, len(rs), 100 * hit / tok, pct([r["response_s"] for r in rs], .5)))
    return out


def main():
    paths = sys.argv[1:]
    runs = [json.loads(Path(p).read_text()) for p in paths]
    H = [headline(d) for d in runs]

    print("%-10s %6s %8s %9s %8s %8s %8s %8s %7s %7s %8s %6s" % (
        "arm", "users", "turns", "makespan", "hit2+%", "recompM",
        "p50", "p90", "p99", "q50", "inflMed", "held"))
    for h in H:
        print("%-10s %6d %8s %8.0fs %8.1f %8.2f %7.1fs %7.1fs %7.1fs %6.1fs %8d %6d" % (
            h["arm"], h["users"], "%d/%d" % (h["turns"], h["all_turns"]),
            h["makespan"], h["hit2"], h["recompute_M"], h["p50"], h["p90"],
            h["p99"], h["q50"], h["infl_med"], h["held"]))
    if len(H) == 2:
        a, b = H
        print("\n%s vs %s:  makespan %.2fx  p50 %.2fx  p90 %.2fx  p99 %.2fx  recompute %.2fx  hit %+.1f pts" % (
            a["arm"], b["arm"], a["makespan"] / b["makespan"], a["p50"] / b["p50"],
            a["p90"] / b["p90"], a["p99"] / b["p99"],
            a["recompute_M"] / b["recompute_M"], b["hit2"] - a["hit2"]))

    w = 200
    print("\nper %ds window: turns2+ done | hit%% | response p50" % w)
    B = [dict((b, (n, h, p)) for b, n, h, p in buckets(d, w)) for d in runs]
    keys = sorted(set().union(*[set(x) for x in B]))
    print("%-12s" % "t" + "".join("%-26s" % d["arm"] for d in runs))
    for k in keys:
        row = "%5d-%-5d " % (k, k + w)
        for x in B:
            if k in x:
                n, h, p = x[k]
                row += "n=%3d %5.1f%% %6.1fs     " % (n, h, p)
            else:
                row += "%-26s" % "-"
        print(row)


if __name__ == "__main__":
    main()
