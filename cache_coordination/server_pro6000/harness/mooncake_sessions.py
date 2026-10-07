"""Mooncake conversation trace -> multi-turn sessions.

Each request carries hash_ids of its 512-token prompt blocks. Turn r+1 of a
conversation repeats all *full* blocks of turn r (the last, partial block is
re-hashed once it fills up), so: parent(j) = latest earlier request whose
full-block list is a prefix of j's hash_ids (>= 2 blocks; block 0 alone is the
system prompt shared by every request).

Writes data/mooncake/conv_sessions.json: {"sessions": [[req_idx, ...], ...]}
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/mooncake/conversation_trace.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/mooncake/conv_sessions.json"
R = [json.loads(l) for l in open(SRC)]
ts = np.array([r["timestamp"] for r in R]) / 1000.0
il = np.array([r["input_length"] for r in R])
ol = np.array([r["output_length"] for r in R])
q = lambda a, p: float(np.percentile(a, p))

idx = {}
parent = [-1] * len(R)
shared = [0] * len(R)
for j, r in enumerate(R):
    h = r["hash_ids"]
    for L in range(len(h) - 1, 1, -1):
        p = idx.get(tuple(h[:L]))
        if p is not None:
            parent[j], shared[j] = p, L
            break
    if len(h) > 1:
        idx[tuple(h[:-1])] = j
child = defaultdict(list)
for j, p in enumerate(parent):
    if p >= 0:
        child[p].append(j)
sess = []
for r0 in [j for j in range(len(R)) if parent[j] == -1]:
    c = [r0]
    while child.get(c[-1]):
        c.append(child[c[-1]][0])
    sess.append(c)
branching = sum(1 for p in child if len(child[p]) > 1)

n = np.array([len(c) for c in sess])
multi = [c for c in sess if len(c) >= 2]
span = ts.max() - ts.min()
print("requests %d over %.0f s (%.2f req/s); sessions %d, multi-turn %d; %.0f%% of requests are in multi-turn sessions; branching parents %d"
      % (len(R), span, len(R) / span, len(sess), len(multi), 100 * sum(len(c) for c in multi) / len(R), branching))
print("turns/session p50 %.0f mean %.2f p90 %.0f p99 %.0f max %d; hist 1..8: %s"
      % (q(n, 50), n.mean(), q(n, 90), q(n, 99), n.max(), np.bincount(np.minimum(n, 8))[1:].tolist()))
gaps = np.array([ts[b] - ts[a] for c in multi for a, b in zip(c, c[1:])])
print("inter-turn gap (submit->submit) s: p10 %.0f p25 %.0f p50 %.0f p75 %.0f p90 %.0f mean %.0f; <=5s %.0f%% <=60s %.0f%%"
      % (q(gaps, 10), q(gaps, 25), q(gaps, 50), q(gaps, 75), q(gaps, 90), gaps.mean(), 100 * (gaps <= 5).mean(), 100 * (gaps <= 60).mean()))
prev_out = np.array([ol[a] for c in multi for a, b in zip(c, c[1:])])
print("previous-turn output at those gaps: p50 %.0f p90 %.0f tokens (Kimi's own service time is inside the gap)" % (q(prev_out, 50), q(prev_out, 90)))
grow = np.array([il[b] - il[a] for c in multi for a, b in zip(c, c[1:])])
print("input growth per turn: p50 %.0f p90 %.0f tokens" % (q(grow, 50), q(grow, 90)))
first = np.array([il[c[0]] for c in multi])
last = np.array([il[c[-1]] for c in multi])
print("multi-turn sessions: first-turn input p50 %.0f mean %.0f p90 %.0f; last-turn input p50 %.0f p90 %.0f max %d"
      % (q(first, 50), first.mean(), q(first, 90), q(last, 50), q(last, 90), last.max()))
sh = np.array([shared[j] * 512 / il[j] for c in multi for j in c[1:]])
print("follow-up turns: share of prompt already computed by previous turn p50 %.2f mean %.2f" % (q(sh, 50), sh.mean()))
st = np.array([ts[c[0]] for c in multi])
en = np.array([ts[c[-1]] for c in multi])
act = [int(((st <= t) & (en >= t)).sum()) for t in np.arange(ts.min(), ts.max(), 60)]
print("multi-turn sessions: %.1f new/min, active per minute mean %.0f max %d" % (len(multi) / (span / 60), np.mean(act), max(act)))
ws = float(sum(il[c[-1]] + ol[c[-1]] for c in multi))
print("KV working set (final ctx of multi-turn sessions): %.1f M tok = %.0f GiB at 144 KiB/tok; per session mean %.0f tok"
      % (ws / 1e6, ws * 147456 / 2 ** 30, ws / len(multi)))
# cross-session sharing beyond the system block: blocks used by >1 session
owner = defaultdict(set)
for k, c in enumerate(sess):
    for j in c:
        for x in R[j]["hash_ids"][1:]:
            owner[x].add(k)
cross = sum(1 for x, s in owner.items() if len(s) > 1)
print("blocks (excluding system block 0) shared across different sessions: %d of %d" % (cross, len(owner)))
json.dump({"sessions": [[int(j) for j in c] for c in sess]}, open(OUT, "w"))
