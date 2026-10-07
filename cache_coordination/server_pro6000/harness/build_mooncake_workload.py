"""Mooncake (Kimi) conversation trace -> closed-loop replay workload.

The trace is one hour of production long-context chat: per request the submit
time, prompt/output length and the hash ids of its 512-token prompt blocks.
Sessions come from conv_sessions.json (mooncake_sessions.py). We keep, per
session: the real start time, the real prompt block structure of every turn
(so prefix sharing within and across sessions is exactly the trace's), the real
output length, and the user think time
    z_r = max(Z_MIN, gap_r - (TTFT_EST + output_r / DECODE_TPS))
i.e. the submit->submit gap minus an estimate of Kimi's own service time.
The harness replays the next turn at (our completion time + z_r).

Load is scaled by sampling a fraction of the sessions (single- and multi-turn
alike, so the mix stays the trace's) whose first request falls in the window.
Token content is synthesised deterministically per hash id at replay time.

usage: build_mooncake_workload.py --frac 0.2 --start 600 --duration 600 --out workloads/mc-f20.json
"""
import argparse
import json
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/mooncake/conversation_trace.jsonl")
    ap.add_argument("--sessions", default="data/mooncake/conv_sessions.json")
    ap.add_argument("--frac", type=float, required=True, help="fraction of sessions to replay")
    ap.add_argument("--start", type=float, default=0.0, help="window start in the trace (s)")
    ap.add_argument("--duration", type=float, default=900.0, help="window length (s)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-ctx", type=int, default=40000, help="drop sessions whose turns exceed this (model max len)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--think-cap", type=float, default=300.0)
    ap.add_argument("--decode-tps", type=float, default=30.0, help="assumed Kimi decode speed for the service estimate")
    ap.add_argument("--ttft-est", type=float, default=2.0)
    ap.add_argument("--z-min", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R = [json.loads(l) for l in open(a.trace)]
    ts = [r["timestamp"] / 1000.0 for r in R]
    S = json.load(open(a.sessions))["sessions"]
    t0 = min(ts)
    inwin = [c for c in S if a.start <= ts[c[0]] - t0 < a.start + a.duration]
    fit, dropped = [], 0
    for c in inwin:
        c = c[:a.max_turns]
        if all(R[j]["input_length"] + R[j]["output_length"] <= a.max_ctx for j in c):
            fit.append(c)
        else:
            dropped += 1
    rng = random.Random("%d:mooncake:%g" % (a.seed, a.frac))
    pick = sorted(rng.sample(fit, int(round(a.frac * len(fit)))), key=lambda c: ts[c[0]])

    sess = []
    for c in pick:
        turns = [{"hash_ids": R[j]["hash_ids"], "input_length": R[j]["input_length"],
                  "output_length": R[j]["output_length"]} for j in c]
        gaps = [ts[b] - ts[a_] for a_, b in zip(c, c[1:])]
        thinks = [min(a.think_cap, max(a.z_min, g - (a.ttft_est + R[a_]["output_length"] / a.decode_tps)))
                  for g, a_ in zip(gaps, c[:-1])]
        sess.append({"start": round(ts[c[0]] - t0 - a.start, 3), "n": len(c), "turns": turns,
                     "thinks": [round(z, 2) for z in thinks], "raw_gaps": [round(g, 2) for g in gaps]})

    n = np.array([s["n"] for s in sess])
    il = np.array([t["input_length"] for s in sess for t in s["turns"]])
    ol = np.array([t["output_length"] for s in sess for t in s["turns"]])
    z = np.array([x for s in sess for x in s["thinks"]])
    q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else float("nan")
    meta = {"kind": "mooncake", "mode": "trace", "source": a.trace, "window": [a.start, a.start + a.duration],
            "horizon_s": a.duration, "frac": a.frac, "seed": a.seed,
            "sessions_in_window": len(inwin), "dropped_over_max_ctx": dropped, "sessions": len(sess),
            "multi_turn_sessions": int((n >= 2).sum()), "turns": int(n.sum()),
            "rate_per_min": 60 * len(sess) / a.duration,
            "input_p50": q(il, 50), "input_mean": float(il.mean()), "input_p90": q(il, 90),
            "output_p50": q(ol, 50), "output_mean": float(ol.mean()),
            "think_p50": q(z, 50), "think_p90": q(z, 90), "think_cap_s": a.think_cap,
            "decode_tps_assumed": a.decode_tps, "ttft_est": a.ttft_est,
            "prompt_tokens_total_M": float(il.sum() / 1e6),
            "final_ctx_working_set_M": float(sum(s["turns"][-1]["input_length"] + s["turns"][-1]["output_length"]
                                                 for s in sess) / 1e6)}
    json.dump({"meta": meta, "sessions": sess}, open(a.out, "w"))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
