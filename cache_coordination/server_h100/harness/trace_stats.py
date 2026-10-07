"""Characterise a Mooncake-format timed trace.

Fields per line: timestamp (ms), input_length, output_length, hash_ids.
hash_ids is the sequence of prefix block hashes, so a shared leading run of ids
between two requests is literally a shared KV prefix. That is what lets us
measure, before running anything, how much reuse the trace offers and whether
its arrival pattern leaves any room for a cache-aware harness.
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

BLOCK = 512  # Mooncake publishes hash ids at 512-token granularity


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def main(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print("%s: not jsonl (first bytes: %r)" % (path, line[:60]))
            return
        if "hash_ids" not in r:
            print("%s: unexpected schema %s" % (path, sorted(r)))
            return
        rows.append(r)

    ins = [r["input_length"] for r in rows]
    outs = [r["output_length"] for r in rows]
    ts = [r["timestamp"] for r in rows]
    span_s = (max(ts) - min(ts)) / 1000.0

    print("=" * 72)
    print("%s   %d requests, %.1f s span, %.1f req/s"
          % (Path(path).name, len(rows), span_s, len(rows) / max(span_s, 1e-9)))
    print("  input_length  p10=%d p50=%d p90=%d max=%d"
          % (pct(ins, 10), pct(ins, 50), pct(ins, 90), max(ins)))
    print("  output_length p10=%d p50=%d p90=%d max=%d"
          % (pct(outs, 10), pct(outs, 50), pct(outs, 90), max(outs)))
    print("  prefill:decode token ratio = %.1f : 1" % (sum(ins) / max(sum(outs), 1)))

    # how much of each request is a prefix some earlier request already had
    seen_prefix = {}          # tuple of hash ids -> first request index
    reuse_tokens = 0
    reuse_hits = 0
    gaps = []
    for i, r in enumerate(rows):
        ids = r["hash_ids"]
        best = 0
        for k in range(len(ids), 0, -1):
            key = tuple(ids[:k])
            if key in seen_prefix:
                best = k
                gaps.append((r["timestamp"] - rows[seen_prefix[key]]["timestamp"]) / 1000.0)
                break
        if best:
            reuse_hits += 1
            reuse_tokens += best * BLOCK
        for k in range(1, len(ids) + 1):
            seen_prefix.setdefault(tuple(ids[:k]), i)

    print("  reusable prefix: %.1f%% of requests, %.1f%% of all input tokens"
          % (100 * reuse_hits / len(rows), 100 * reuse_tokens / sum(ins)))
    if gaps:
        print("  gap to the request that established the prefix: "
              "p10=%.1fs p50=%.1fs p90=%.1fs"
              % (pct(gaps, 10), pct(gaps, 50), pct(gaps, 90)))

    # conversation structure: how many requests hang off each root block
    roots = Counter(r["hash_ids"][0] for r in rows)
    chains = defaultdict(list)
    for r in rows:
        chains[r["hash_ids"][0]].append(r)
    depths = [len(v) for v in chains.values()]
    print("  %d distinct root blocks, requests per root: med=%d max=%d"
          % (len(roots), int(statistics.median(depths)), max(depths)))

    kv_gib = sum(ins) * 144 / 1024 / 1024
    uniq_gib = len(seen_prefix and set(
        h for r in rows for h in r["hash_ids"])) * BLOCK * 144 / 1024 / 1024
    print("  KV if nothing were shared: %.1f GiB; unique blocks: %.1f GiB"
          % (kv_gib, uniq_gib))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
