"""BurstGPT_3.csv -> per-session structure for trace-driven replay.

Columns: Timestamp (s from day-1 00:00), Session ID, Elapsed time (s, submit ->
full response), Model, Request/Response/Total tokens, Log Type.

For every conversation session u with turns r = 1..n (sorted by submit time):
    z_{u,r} = max(0, tau_{u,r+1} - (tau_{u,r} + e_{u,r}))      user think time
A gap longer than SPLIT_S is treated as a new session (user came back later).

Outputs data/burstgpt/sessions.json:
    {"sessions": [{"start": s_u, "thinks": [z_1..z_{n-1}], "n": n, "model": m,
                   "req_tokens": [...], "resp_tokens": [...]}, ...],
     "stats": {...}}
and prints the numbers needed to design the replay (turn distribution, think
time quantiles, session-start rate per hour of day, busiest hours).
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/burstgpt/BurstGPT_3.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/burstgpt/sessions.json"
SPLIT_S = 30 * 60          # a >30 min silence ends the session

by = defaultdict(list)
n_rows = n_conv = 0
with open(SRC, newline="") as f:
    for row in csv.DictReader(f):
        n_rows += 1
        if row["Log Type"] != "Conversation log" or not row["Session ID"]:
            continue
        n_conv += 1
        try:
            by[row["Session ID"]].append((float(row["Timestamp"]), float(row["Elapsed time"]),
                                          row["Model"], int(row["Request tokens"]),
                                          int(row["Response tokens"])))
        except ValueError:
            continue

sessions = []
for sid, rows in by.items():
    rows.sort()
    cur = [rows[0]]
    for prev, nxt in zip(rows, rows[1:]):
        gap = nxt[0] - (prev[0] + prev[1])
        if gap > SPLIT_S:
            sessions.append(cur)
            cur = []
        cur.append(nxt)
    sessions.append(cur)

recs = []
for s in sessions:
    thinks = [max(0.0, b[0] - (a[0] + a[1])) for a, b in zip(s, s[1:])]
    recs.append({"start": s[0][0], "n": len(s), "thinks": [round(z, 2) for z in thinks],
                 "model": s[0][2], "req_tokens": [r[3] for r in s], "resp_tokens": [r[4] for r in s],
                 "elapsed": [round(r[1], 2) for r in s]})

n = np.array([r["n"] for r in recs])
multi = [r for r in recs if r["n"] >= 2]
z = np.array([t for r in multi for t in r["thinks"]])
q = lambda a, p: float(np.percentile(a, p))
starts = np.array([r["start"] for r in multi])
hour = ((starts % 86400) // 3600).astype(int)
day = (starts // 86400).astype(int)
per_hour = np.bincount(hour, minlength=24) / max(1, len(set(day)))       # multi-turn sessions/hour of day, per day
# busiest single hours (day, hour) by number of multi-turn session starts
dh = defaultdict(int)
for d_, h_ in zip(day, hour):
    dh[(int(d_), int(h_))] += 1
busiest = sorted(dh.items(), key=lambda kv: -kv[1])[:8]

stats = {
    "rows": n_rows, "conversation_rows": n_conv, "sessions_total": len(recs),
    "sessions_multi_turn": len(multi), "days": int(len(set(day))),
    "turns_per_session": {"mean": float(n.mean()), "p50": q(n, 50), "p90": q(n, 90), "p99": q(n, 99),
                          "max": int(n.max()), "hist_1_to_10": np.bincount(np.minimum(n, 10), minlength=11)[1:].tolist()},
    "turns_multi_only": {"mean": float(np.mean([r["n"] for r in multi])), "p50": q([r["n"] for r in multi], 50),
                         "p90": q([r["n"] for r in multi], 90)},
    "think_s": {"mean": float(z.mean()), "p10": q(z, 10), "p25": q(z, 25), "p50": q(z, 50), "p75": q(z, 75),
                "p90": q(z, 90), "p99": q(z, 99), "frac_le_5s": float((z <= 5).mean()),
                "frac_le_60s": float((z <= 60).mean())},
    "elapsed_s": {"p50": q([e for r in multi for e in r["elapsed"]], 50),
                  "p90": q([e for r in multi for e in r["elapsed"]], 90)},
    "req_tokens": {"p50": q([t for r in multi for t in r["req_tokens"]], 50),
                   "p90": q([t for r in multi for t in r["req_tokens"]], 90)},
    "multi_sessions_per_hour_of_day_avg": [round(float(x), 1) for x in per_hour],
    "busiest_day_hours": [{"day": d_, "hour": h_, "sessions": c} for (d_, h_), c in busiest],
}
json.dump({"sessions": recs, "stats": stats, "split_s": SPLIT_S}, open(OUT, "w"))
print(json.dumps(stats, indent=1))
