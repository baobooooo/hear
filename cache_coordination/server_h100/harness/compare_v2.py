"""Acceptance check against native FCFS baseline on the same seed.

Pass criteria (frozen):
    mean TTFT  <= baseline      (all turns)
    P95  TTFT  <= baseline      (all turns)
    max  TTFT  <= baseline      (the slowest wait may not exceed baseline's slowest)
    makespan   <  baseline      (whole batch: first arrival -> last token)
Reported alongside, not part of the pass/fail:
    first-turn TTFT mean/P95/max (the turns most exposed to queue-jumping),
    follow-up TTFT, decode time, local prompt compute, tokens reusable at
    arrival but lost before execution, harness wait, session time.

usage: compare_v2.py <baseline.json> <candidate.json> [<candidate.json> ...]
"""

import json
import sys
from collections import defaultdict


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def mean(xs):
    return sum(xs) / len(xs)


def ttft(d):
    return max(d["decode_start"], d["t_arrive"]) - d["t_arrive"]


def summary(path):
    run = json.load(open(path))
    R = run["records"]
    first = [d for d in R if d["r"] == 0]
    later = [d for d in R if d["r"] > 0]
    by = defaultdict(list)
    for d in R:
        by[d["inst"]].append(d)
    sess = [max(x["decode_end"] for x in v) - min(x["t_arrive"] for x in v) for v in by.values()]
    tt = [ttft(d) for d in R]
    e2e = [d["decode_end"] - d["t_arrive"] for d in R]
    has_arr = "arr_gpu" in R[0]
    return {
        "label": "%s seed=%s%s" % (run["group"], run.get("seed"),
                                   " k=%s g=%s" % (run.get("k"), run.get("guard")) if run["group"] in ("sched", "tiered") else ""),
        "ttft_mean": mean(tt), "ttft_p95": q(tt, .95), "ttft_p99": q(tt, .99), "ttft_max": max(tt),
        "first_mean": mean([ttft(d) for d in first]), "first_p95": q([ttft(d) for d in first], .95),
        "first_max": max(ttft(d) for d in first),
        "follow_mean": mean([ttft(d) for d in later]), "follow_p95": q([ttft(d) for d in later], .95),
        "e2e_mean": mean(e2e), "e2e_p95": q(e2e, .95),
        "decode_mean": mean([(d["gen_ms"] or 0) / 1000 for d in R]),
        "local_M": sum(d["prompt_len"] - d["cached"] for d in R) / 1e6,
        "lost_M": (sum(max(0, d["arr_gpu"] + d["arr_cpu"] - d["cached"]) for d in R) / 1e6
                   if has_arr else float("nan")),
        "hwait_mean": mean([d.get("harness_wait_s", d["t_send"] - d["t_arrive"]) for d in R]),
        "sess_mean": mean(sess), "sess_p95": q(sess, .95),
        "makespan": max(d["decode_end"] for d in R),
        "first": {d["inst"]: ttft(d) for d in first},
    }


ROWS = [
    ("TTFT mean (all)", "ttft_mean", True), ("TTFT P95 (all)", "ttft_p95", True),
    ("TTFT max (all)", "ttft_max", True),
    ("makespan (batch E2E)", "makespan", True),
    (None, None, None),
    ("TTFT P99 (all)", "ttft_p99", False),
    ("first-turn TTFT mean", "first_mean", False), ("first-turn TTFT P95", "first_p95", False),
    ("first-turn TTFT max", "first_max", False),
    ("follow-up TTFT mean", "follow_mean", False), ("follow-up TTFT P95", "follow_p95", False),
    ("request E2E mean", "e2e_mean", False), ("request E2E P95", "e2e_p95", False),
    ("decode mean", "decode_mean", False),
    ("local prompt compute (M tok)", "local_M", False),
    ("reusable@arrival lost (M tok)", "lost_M", False),
    ("harness wait mean", "hwait_mean", False),
    ("session time mean", "sess_mean", False), ("session time P95", "sess_p95", False),
]


def main():
    base = summary(sys.argv[1])
    cands = [summary(p) for p in sys.argv[2:]]
    w = 20
    print("%-32s %s" % ("", "".join("%*s" % (w, s["label"][:w - 1]) for s in [base] + cands)))
    for name, key, gate in ROWS:
        if name is None:
            print("-" * (32 + w * (1 + len(cands))))
            continue
        cells = "%*.2f" % (w, base[key])
        for c in cands:
            ratio = base[key] / c[key] if c[key] else float("nan")
            mark = ""
            if gate:
                ok = c[key] < base[key] if key == "makespan" else c[key] <= base[key]
                mark = " PASS" if ok else " FAIL"
            cells += "%*s" % (w, "%.2f (%.2fx)%s" % (c[key], ratio, mark))
        print("%-32s %s" % (name + (" *" if gate else ""), cells))
    print("\n* = acceptance gate. ratio = baseline / candidate (> 1 means candidate is better).")
    for c in cands:
        common = [i for i in base["first"] if i in c["first"]]
        worse = sorted((c["first"][i] - base["first"][i] for i in common), reverse=True)
        n_worse = sum(1 for x in worse if x > 0.5)
        print("%s: first turns slower than baseline by > 0.5 s: %d/%d, worst %s"
              % (c["label"], n_worse, len(common), ", ".join("+%.1f" % x for x in worse[:3])))


if __name__ == "__main__":
    main()
