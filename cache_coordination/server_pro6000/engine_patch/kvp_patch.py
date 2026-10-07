"""Scheduler-side patches for vLLM 0.26 (priority policy + CPU offloading).

KVP_PROTECT=1  -- deadline protection inside the engine.
    The harness sends `vllm_xargs: {"protect_s": W}` with a request. Once the
    request has waited W seconds in the engine's waiting queue it is promoted:
    its priority becomes PROMO + deadline_ms, which sorts before every normal
    priority, so nobody can be admitted ahead of it any more (earliest deadline
    first among promoted requests). Promoted requests also become the last
    candidates for priority preemption. Requests without protect_s are never
    promoted.

KVP_KEEPALIVE=1 -- keep the CPU (L2) copies of queued requests alive.
    Every KVP_KEEPALIVE_S seconds, after a scheduling step, the offload
    manager's LRU is touched in this order: running requests, then waiting
    requests from the back of the queue to the head. The head of the queue is
    therefore the most-recently-used entry and the last to be evicted; blocks
    no queued or running request refers to are evicted first. Touch only
    reorders the LRU -- it never blocks an eviction.

KVP_RETAIN=1 -- session-aware L2 retention for *finished* requests.
    The harness sends `vllm_xargs: {"p_return": p}`: its estimate that the
    user will come back for another turn (e.g. from the session depth). When a
    request finishes, its L2 chunks are remembered together with p and the
    finish time. On every keep-alive tick the remembered contexts are ranked by
        value = p * exp(-idle / KVP_RETAIN_TAU)
    and the top ones that fit in the L2 (by chunk count) are touched in
    ascending value order, so the most valuable context is the most recently
    used and the last to be evicted; contexts of sessions that are unlikely to
    return (or have been idle for long) fall to the LRU end and go first.
    Queued/running requests are touched after them and stay on top.

KVP_FEEDBACK=1 -- per-request status back to the client (needs the offloading
    connector). The completion response's `kv_transfer_params` carries
    {kvp_promoted, kvp_wait_s, kvp_protect_s, kvp_l1_tokens, kvp_l2_tokens}.
"""
import heapq
import logging
import math
import os
import time

log = logging.getLogger("vllm.kvp_patch")

PROMO = -10**15                 # deadline_ms (~1.8e12) keeps PROMO + d < -9e14
PROMO_LIMIT = -10**14
STATS = {"promoted": 0, "keepalive_runs": 0, "keepalive_reqs": 0, "keepalive_s": 0.0,
         "retained": 0, "retain_touched": 0, "retain_dropped": 0}
REQ: dict = {}                  # request_id -> per-request status (feedback)
RETAIN: dict = {}               # request_id -> {"keys", "t", "p"} for finished requests
_last = {"keepalive": 0.0, "log": 0.0}


def _xargs(req):
    sp = getattr(req, "sampling_params", None)
    return (getattr(sp, "extra_args", None) or {}) if sp is not None else {}


def _deadline(req):
    d = getattr(req, "_kvp_deadline", None)
    if d is None:
        w = _xargs(req).get("protect_s")
        d = req.arrival_time + float(w) if w is not None else float("inf")
        req._kvp_deadline = d
        req._kvp_protect_s = w
    return d


def _promote(queue):
    heap = queue._heap
    if not heap:
        return
    now = time.time()
    changed = False
    for r in heap:
        if r.priority > PROMO_LIMIT:
            d = _deadline(r)
            if d <= now:
                r.priority = PROMO + int(d * 1000)
                STATS["promoted"] += 1
                changed = True
                rec = REQ.setdefault(r.request_id, {})
                rec["kvp_promoted"] = True
                rec["kvp_promoted_wait_s"] = round(now - r.arrival_time, 3)
    if changed:
        heapq.heapify(heap)


def _retain_touch(cs, now):
    """Rank remembered finished contexts by value and re-touch the ones that fit."""
    tau = float(os.environ.get("KVP_RETAIN_TAU", "400"))
    max_age = float(os.environ.get("KVP_RETAIN_MAX_AGE", "1800"))
    policy = getattr(cs.manager, "_policy", None)
    cap = getattr(cs.manager, "_num_blocks", 0) or 0
    if policy is None or cap <= 0:
        return
    ranked = []
    for rid, rec in list(RETAIN.items()):
        age = now - rec["t"]
        keys = rec["keys"]
        # forget a context once its chunks have left the L2 (probe the middle and
        # the last stored chunk; the shared first chunk would always be present)
        gone = policy.get(keys[-1]) is None and policy.get(keys[len(keys) // 2]) is None
        if age > max_age or gone:
            RETAIN.pop(rid, None)
            STATS["retain_dropped"] += 1
            continue
        ranked.append((rec["p"] * math.exp(-age / tau), keys))
    ranked.sort(key=lambda x: -x[0])
    keep, n = [], 0
    for v, keys in ranked:                      # highest value first, until the L2 is full
        if n + len(keys) > cap:
            break
        keep.append(keys)
        n += len(keys)
    for keys in reversed(keep):                 # ascending value: best one is touched last (MRU)
        cs.manager.touch(keys, None)
    STATS["retain_touched"] += len(keep)
    STATS["retained"] = len(RETAIN)


def _keepalive(sched, retain):
    now = time.monotonic()
    if now - _last["keepalive"] < float(os.environ.get("KVP_KEEPALIVE_S", "0.5")):
        return
    _last["keepalive"] = now
    cs = getattr(sched.connector, "connector_scheduler", None) if sched.connector else None
    if cs is None:
        return
    t0 = time.perf_counter()
    if retain:
        # load-aware: when the engine is overloaded (long waiting queue) the GPU
        # pool is the bottleneck and pulling more L2 contexts back only causes
        # preemptions, so retention is paused until the queue drains
        max_wait = int(os.environ.get("KVP_RETAIN_MAX_WAIT", "60"))
        if len(sched.waiting) + len(sched.skipped_waiting) <= max_wait:
            _retain_touch(cs, time.time())
        else:
            STATS["retain_paused"] = STATS.get("retain_paused", 0) + 1
    status = cs._req_status
    n = 0
    for req in list(sched.running):
        st = status.get(req.request_id)
        if st is not None:
            cs._touch(st)
            n += 1
    waiting = list(sched.waiting) + list(sched.skipped_waiting)
    waiting.sort()                          # head of queue first
    for req in reversed(waiting):           # back of queue touched first
        st = status.get(req.request_id)
        if st is None:
            continue
        if not st.transfer_jobs:
            st.update_offload_keys()
        cs._touch(st)
        n += 1
    STATS["keepalive_runs"] += 1
    STATS["keepalive_reqs"] += n
    STATS["keepalive_s"] += time.perf_counter() - t0


def apply():
    from vllm.v1.core.sched import request_queue as rq
    from vllm.v1.core.sched import scheduler as sm

    protect = os.environ.get("KVP_PROTECT") == "1"
    keepalive = os.environ.get("KVP_KEEPALIVE") == "1"
    feedback = os.environ.get("KVP_FEEDBACK") == "1"
    retain = os.environ.get("KVP_RETAIN") == "1"

    if protect:
        P = rq.PriorityRequestQueue
        orig_peek, orig_pop = P.peek_request, P.pop_request

        def peek_request(self):
            _promote(self)
            return orig_peek(self)

        def pop_request(self):
            _promote(self)
            return orig_pop(self)

        P.peek_request = peek_request
        P.pop_request = pop_request

    if feedback or retain:
        from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector as oc
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as ocs

        if feedback:
            orig_match = ocs.OffloadingConnectorScheduler.get_num_new_matched_tokens

            def get_num_new_matched_tokens(self, request, num_computed_tokens):
                out = orig_match(self, request, num_computed_tokens)
                try:
                    rec = REQ.setdefault(request.request_id, {})
                    rec["kvp_l1_tokens"] = int(num_computed_tokens)
                    rec["kvp_l2_tokens"] = int(out[0] or 0)
                    rec["kvp_wait_s"] = round(time.time() - request.arrival_time, 3)
                    rec.setdefault("kvp_protect_s", getattr(request, "_kvp_protect_s", None))
                    rec.setdefault("kvp_promoted", False)
                except Exception:
                    log.exception("kvp feedback (match) failed")
                return out

            ocs.OffloadingConnectorScheduler.get_num_new_matched_tokens = get_num_new_matched_tokens

        # the scheduler calls one of these two, depending on the KV-group layout
        for name in ("request_finished", "request_finished_all_groups"):
            orig_finished = getattr(oc.OffloadingConnector, name, None)
            if orig_finished is None:
                continue

            def make(orig):
                def wrapper(self, request, block_ids):
                    if retain:
                        try:
                            p = _xargs(request).get("p_return")
                            st = self.connector_scheduler._req_status.get(request.request_id)
                            if p is not None and st is not None:
                                # only chunks that are actually stored in the L2 (prompt-only
                                # offloading never stores the trailing decode chunks)
                                pol = getattr(self.connector_scheduler.manager, "_policy", None)
                                keys = [k for g in st.group_states for k in g.offload_keys
                                        if pol is None or pol.get(k) is not None]
                                if len(keys) >= 2:
                                    RETAIN[request.request_id] = {"keys": keys, "t": time.time(), "p": float(p)}
                                    STATS["retain_stored_chunks"] = STATS.get("retain_stored_chunks", 0) + len(keys)
                        except Exception:
                            log.exception("kvp retain (finish) failed")
                    delay, params = orig(self, request, block_ids)
                    if feedback:
                        rec = REQ.pop(request.request_id, None)
                        if rec:
                            params = {**(params or {}), **rec}
                    return delay, params
                return wrapper

            setattr(oc.OffloadingConnector, name, make(orig_finished))

    orig_schedule = sm.Scheduler.schedule

    def schedule(self, *a, **k):
        out = orig_schedule(self, *a, **k)
        if keepalive or retain:
            try:
                _keepalive(self, retain)
            except Exception:                # never take the engine down
                log.exception("kvp keepalive failed")
        now = time.monotonic()
        if now - _last["log"] > 30:
            _last["log"] = now
            log.warning("KVP stats %s", STATS)
        return out

    sm.Scheduler.schedule = schedule
    log.warning("KVP patch applied: protect=%s keepalive=%s feedback=%s retain=%s",
                protect, keepalive, feedback, retain)
