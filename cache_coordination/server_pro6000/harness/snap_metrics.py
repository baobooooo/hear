"""Snapshot the engine's cache counters, or diff two snapshots.

L1 hits  = vllm:prefix_cache_hits_total           (GPU prefix cache, tokens)
L2 hits  = vllm:external_prefix_cache_hits_total  (restored by the offloading connector)
traffic  = vllm:kv_offload_total_bytes_total / _time_total, per label set

usage:
    snap_metrics.py snap <base_url> <out.json>
    snap_metrics.py diff <before.json> <after.json>
"""

import json
import re
import sys
import urllib.request

KEEP = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total",
        "vllm:kv_offload_total_bytes_total", "vllm:kv_offload_total_time_total",
        "vllm:num_preemptions_total")
LINE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.e+-]+)$")


def snap(base, out):
    url = base.rstrip("/").rsplit("/v1", 1)[0] + "/metrics"
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    vals = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(1) in KEEP:
            labels = re.sub(r'(engine|model_name)="[^"]*",?', "", m.group(2) or "").strip("{},")
            key = m.group(1) + ("{" + labels + "}" if labels else "")
            vals[key] = vals.get(key, 0.0) + float(m.group(3))
    json.dump(vals, open(out, "w"), indent=1)


def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    d = {k: B.get(k, 0) - A.get(k, 0) for k in set(A) | set(B)}
    l1q = d.get("vllm:prefix_cache_queries_total", 0)
    l1h = d.get("vllm:prefix_cache_hits_total", 0)
    l2q = d.get("vllm:external_prefix_cache_queries_total", 0)
    l2h = d.get("vllm:external_prefix_cache_hits_total", 0)
    print("prompt tokens queried : %.2fM" % (l1q / 1e6))
    print("L1 hit                : %.2fM tokens = %.1f%% of prompt tokens" % (l1h / 1e6, 100 * l1h / max(l1q, 1)))
    print("L2 hit (restored)     : %.2fM tokens = %.1f%% of prompt tokens  (%.1f%% of what L1 missed)"
          % (l2h / 1e6, 100 * l2h / max(l1q, 1), 100 * l2h / max(l2q, 1)))
    print("recomputed            : %.2fM tokens = %.1f%%" % ((l1q - l1h - l2h) / 1e6, 100 * (l1q - l1h - l2h) / max(l1q, 1)))
    for k in sorted(d):
        if k.startswith("vllm:kv_offload") and d[k]:
            print("  %-60s %.3g" % (k, d[k]))
    print("preemptions           : %d" % d.get("vllm:num_preemptions_total", 0))


if __name__ == "__main__":
    if sys.argv[1] == "snap":
        snap(sys.argv[2], sys.argv[3])
    else:
        diff(sys.argv[2], sys.argv[3])
