"""Cross-process prefill admission gate for chain-cached sparse endpoints.

SnapKV compacts a chain only when its prefill finishes, so every in-flight
prefill reserves its whole suffix in the KV pool.  When the pool is full,
Sparse-vLLM admits the next request by evicting the least recently used IDLE
chains, and the victims are exactly the sub-agents that finished a round
early and are about to resume.  Losing a ~5k-slot compressed chain turns the
next round's ~45k-token delta into a ~150k-token full prefill.

The gate keeps the sum of in-flight prefill reservations below a token budget
(pool minus room for every live compressed chain), so the engine never needs
to evict.  Requests wait here in FIFO order instead.  A grant is returned as
soon as the first token streams back, which is when the engine clears the
prefill reservation.

Every workflow instance is its own process, so the state lives in a JSON file
guarded by flock.  The gate is off unless MTBENCH_PREFILL_GATE_FILE and
MTBENCH_PREFILL_GATE_TOKENS are both set.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass

POLL_SECONDS = 0.25


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class Grant:
    grant_id: str
    tokens: int
    waited_seconds: float
    in_flight_tokens: int


class PrefillGate:
    def __init__(self, path: str, budget_tokens: int):
        self.path = path
        self.budget = int(budget_tokens)

    @classmethod
    def from_env(cls, role: str) -> "PrefillGate | None":
        path = os.environ.get("MTBENCH_PREFILL_GATE_FILE")
        budget = os.environ.get("MTBENCH_PREFILL_GATE_TOKENS")
        gated_role = os.environ.get("MTBENCH_PREFILL_GATE_ROLE", "researcher")
        if not path or not budget or role != gated_role:
            return None
        return cls(path, int(budget))

    @classmethod
    def request_gate_from_env(cls, role: str) -> "PrefillGate | None":
        """Cap how many of this role's requests run at once, across processes.

        Every workflow instance is its own process, so each one only sees its
        own main-agent request; six instances crossing a round barrier together
        put six long requests on the main GPU at once and each one slows down.
        This gate costs one slot per request and holds it until the request
        finishes, so the arms can be compared at the same main-agent batch size.
        """
        path = os.environ.get("MTBENCH_MAIN_GATE_FILE")
        slots = os.environ.get("MTBENCH_MAIN_GATE_SLOTS")
        gated_role = os.environ.get("MTBENCH_MAIN_GATE_ROLE", "main")
        if not path or not slots or role != gated_role:
            return None
        return cls(path, int(slots))

    def _update(self, mutate):
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read()
                state = json.loads(raw) if raw.strip() else {}
                holders = {
                    key: value
                    for key, value in state.get("holders", {}).items()
                    if _alive(int(value["pid"]))
                }
                queue = [
                    entry for entry in state.get("queue", [])
                    if _alive(int(entry["pid"]))
                ]
                state = {"holders": holders, "queue": queue}
                result = mutate(state)
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(state))
                handle.flush()
                return result
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _enqueue(self, grant_id: str) -> None:
        def mutate(state):
            state["queue"].append({"id": grant_id, "pid": os.getpid()})
        self._update(mutate)

    def _try_acquire(self, grant_id: str, tokens: int) -> int | None:
        def mutate(state):
            queue = state["queue"]
            if not queue or queue[0]["id"] != grant_id:
                return None
            in_flight = sum(int(value["tokens"]) for value in state["holders"].values())
            # A request larger than the whole budget still runs once it is alone.
            if state["holders"] and in_flight + tokens > self.budget:
                return None
            queue.pop(0)
            state["holders"][grant_id] = {
                "tokens": int(tokens), "pid": os.getpid(), "t": time.time(),
            }
            return in_flight
        return self._update(mutate)

    def _forget(self, grant_id: str) -> None:
        def mutate(state):
            state["holders"].pop(grant_id, None)
            state["queue"] = [e for e in state["queue"] if e["id"] != grant_id]
        self._update(mutate)

    async def acquire(self, tokens: int) -> Grant:
        grant_id = uuid.uuid4().hex
        started = time.perf_counter()
        await asyncio.to_thread(self._enqueue, grant_id)
        try:
            while True:
                in_flight = await asyncio.to_thread(self._try_acquire, grant_id, tokens)
                if in_flight is not None:
                    return Grant(grant_id, int(tokens), time.perf_counter() - started, in_flight)
                await asyncio.sleep(POLL_SECONDS)
        except BaseException:
            await asyncio.to_thread(self._forget, grant_id)
            raise

    async def release(self, grant: Grant) -> None:
        await asyncio.to_thread(self._forget, grant.grant_id)
