"""Time series of what the GPU and the KV pool are doing during a run.

Every INTERVAL seconds: the engine's own counters (/metrics) plus the card's
utilisation from nvidia-smi. Written into the run JSON as "samples", so the
hit-rate curve, the L1 occupancy curve and GPU utilisation can be read off the
same timeline as the per-turn Gantt.

Fields per sample:
    t                 seconds since the harness clock zero
    util, mem_used    GPU SM utilisation (%) and memory in use (MiB)
    usage             L1 KV pool occupancy (0..1)
    running, waiting  requests in the engine
    preemptions       cumulative
    hits, queries     cumulative engine prefix-cache counters (L1)
    l2_in, l2_out     cumulative CPU<->GPU offload bytes
"""
import re
import subprocess
import threading
import time
import urllib.request

LINE = re.compile(r"^(vllm:[a-z_0-9]+)(?:\{([^}]*)\})?\s+([0-9.e+-]+)$")
WANTED = {
    "vllm:kv_cache_usage_perc": "usage",
    "vllm:gpu_cache_usage_perc": "usage",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:prefix_cache_hits_total": "hits",
    "vllm:prefix_cache_queries_total": "queries",
    "vllm:external_prefix_cache_hits_total": "ext_hits",
    "vllm:external_prefix_cache_queries_total": "ext_queries",
}
OFFLOAD = "vllm:kv_offload_total_bytes_total"


class Sampler(threading.Thread):
    def __init__(self, base_url, gpu, clock, interval=0.5):
        super().__init__(daemon=True)
        self.url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/metrics"
        self.gpu = str(gpu)
        self.clock = clock              # callable -> seconds since run start
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def _gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--id=" + self.gpu, "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            util, mem = out.strip().split(",")
            return float(util), float(mem)
        except Exception:
            return float("nan"), float("nan")

    def _metrics(self):
        out = {}
        try:
            with urllib.request.urlopen(self.url, timeout=5) as r:
                text = r.read().decode()
        except Exception:
            return out
        for line in text.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
            if name in WANTED:
                out[WANTED[name]] = val
            elif name == OFFLOAD:
                if 'transfer_type="CPU_to_GPU"' in labels:
                    out["l2_in"] = val
                elif 'transfer_type="GPU_to_CPU"' in labels:
                    out["l2_out"] = val
        return out

    def run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            util, mem = self._gpu()
            s = {"t": round(self.clock(), 2), "util": util, "mem_used": mem}
            s.update(self._metrics())
            self.samples.append(s)
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))

    def stop(self):
        self._stop.set()
