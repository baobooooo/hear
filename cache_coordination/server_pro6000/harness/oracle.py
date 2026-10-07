"""CacheOracle: a harness-side mirror of what KV the engine currently holds.

vLLM publishes BlockStored / BlockRemoved on a ZMQ PUB socket, one stream for
both tiers, tagged with medium="GPU" (L1) or medium="CPU" (L2). Stores carry
block_hashes, the parent block hash, and the token_ids the blocks cover, so we
can rebuild the engine's prefix tree *by content* and never have to reimplement
vLLM's block hashing:

    edges[(parent_hash, tokens_of_this_block)] -> block_hash
    residency[block_hash] -> {"GPU", "CPU"}

Matching a candidate prompt is then a walk from the root, one block at a time,
which yields how many of its leading tokens the engine can serve and from which
tier. Note the CPU tier announces at chunk granularity (16 blocks), so L2
matches move in 256-token steps.
"""

import threading
import time
from collections import defaultdict

import msgspec
import zmq


class CacheOracle:
    def __init__(self, endpoint="tcp://127.0.0.1:5557", topic="", block_size=16):
        self.endpoint = endpoint
        self.topic = topic
        self.block_size = block_size
        self._edges: dict[tuple, int] = {}
        self._residency: dict[int, set] = defaultdict(set)
        self._last_seen: dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.stats = {"batches": 0, "stored": 0, "removed": 0}

    # ------------------------------------------------------------------ feed
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.endpoint)
        sub.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
        dec = msgspec.msgpack.Decoder()
        while not self._stop.is_set():
            if not sub.poll(200):
                continue
            _t, _s, payload = sub.recv_multipart()
            batch = dec.decode(payload)
            now = time.time()
            with self._lock:
                self.stats["batches"] += 1
                for ev in batch[1]:
                    if not isinstance(ev, dict):
                        continue
                    kind = ev.get("type")
                    medium = ev.get("medium", "GPU")
                    hashes = ev.get("block_hashes") or []
                    if kind == "BlockStored":
                        self.stats["stored"] += len(hashes)
                        toks = ev.get("token_ids") or []
                        bs = ev.get("block_size") or self.block_size
                        parent = ev.get("parent_block_hash")
                        for i, h in enumerate(hashes):
                            chunk = tuple(toks[i * bs:(i + 1) * bs])
                            if len(chunk) == bs:
                                self._edges[(parent, chunk)] = h
                            self._residency[h].add(medium)
                            self._last_seen[h] = now
                            parent = h
                    elif kind == "BlockRemoved":
                        self.stats["removed"] += len(hashes)
                        for h in hashes:
                            self._residency[h].discard(medium)
                    elif kind == "AllBlocksCleared":
                        self._residency.clear()

    # --------------------------------------------------------------- queries
    def match(self, token_ids):
        """Return (gpu_tokens, cpu_tokens, total_matched_tokens) for a prompt.

        gpu_tokens is the leading run served from L1; total_matched extends that
        with blocks that are only in L2. Anything past the first missing block
        has to be recomputed, so we stop there.
        """
        bs = self.block_size
        parent = None
        gpu = 0
        total = 0
        still_gpu = True
        with self._lock:
            n_blocks = len(token_ids) // bs
            for i in range(n_blocks):
                chunk = tuple(token_ids[i * bs:(i + 1) * bs])
                h = self._edges.get((parent, chunk))
                if h is None:
                    break
                where = self._residency.get(h)
                if not where:
                    break
                total += bs
                if still_gpu and "GPU" in where:
                    gpu += bs
                else:
                    still_gpu = False
                parent = h
        return gpu, total - gpu, total

    def snapshot(self):
        with self._lock:
            gpu = sum(1 for w in self._residency.values() if "GPU" in w)
            cpu = sum(1 for w in self._residency.values() if "CPU" in w)
            return {"blocks_gpu": gpu, "blocks_cpu": cpu,
                    "edges": len(self._edges), **self.stats}
