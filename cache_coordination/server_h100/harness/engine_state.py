"""Poll the engine's own view of its KV pool, from /metrics.

The oracle says *what* is cached; these counters say *how full* the pool is and
whether the engine is thrashing. Together they let the harness size the next
wave instead of guessing:

    vllm:kv_cache_usage_perc                          live L1 occupancy
    vllm:num_requests_waiting_by_reason{capacity}     blocked on blocks
    vllm:num_preemptions_total                        thrash indicator
    vllm:prefix_cache_hits_total / queries_total      realised hit rate
"""

import re
import urllib.request

WANTED = {
    "vllm:kv_cache_usage_perc": "usage",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:prefix_cache_hits_total": "hits",
    "vllm:prefix_cache_queries_total": "queries",
}
LINE = re.compile(r"^(vllm:[a-z_]+)\{([^}]*)\}\s+([0-9.e+-]+)$")


class EngineState:
    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/metrics"
        self.last = {k: 0.0 for k in WANTED.values()}
        self.last["waiting_capacity"] = 0.0

    def sample(self) -> dict:
        try:
            with urllib.request.urlopen(self.url, timeout=5) as r:
                text = r.read().decode()
        except Exception:
            return self.last
        out = {}
        for line in text.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2), float(m.group(3))
            if name in WANTED:
                out[WANTED[name]] = val
            elif (name == "vllm:num_requests_waiting_by_reason"
                  and 'reason="capacity"' in labels):
                out["waiting_capacity"] = val
        self.last = {**self.last, **out}
        return self.last
