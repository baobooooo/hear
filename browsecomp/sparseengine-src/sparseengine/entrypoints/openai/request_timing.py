"""CPU wall-clock observations at the existing dispatcher token boundary."""
from dataclasses import dataclass, field
import time


@dataclass
class RequestTiming:
    submitted_at: float = field(default_factory=time.perf_counter)
    queued_at: float | None = None
    admitted_at: float | None = None
    scheduled_at: float | None = None
    prefill_steps: int = 0
    decode_steps: int = 0
    token_events: list[tuple[float, int]] = field(default_factory=list)

    def scheduled(self, is_prefill: bool, now: float | None = None) -> None:
        if self.scheduled_at is None:
            self.scheduled_at = time.perf_counter() if now is None else now
        if is_prefill:
            self.prefill_steps += 1
        else:
            self.decode_steps += 1

    def observe(self, count: int, now: float | None = None) -> None:
        if count > 0:
            self.token_events.append((time.perf_counter() if now is None else now, count))

    def metrics(self, request_started_at: float, output_tokens: int) -> dict:
        events = self.token_events
        observed = sum(count for _, count in events)
        coalesced = sum(count != 1 for _, count in events)
        complete = bool(events) and observed == output_tokens and coalesced == 0
        first = events[0][0] if events else None
        last = events[-1][0] if events else None
        gaps = [(b[0] - a[0]) * 1000 for a, b in zip(events, events[1:])]
        ordered = all(gap >= 0 for gap in gaps) and (first is None or first >= request_started_at)
        complete = complete and ordered
        decode_ms = (last - first) * 1000 if complete else None
        return {
            "schema": "bcgraph.server_timing.v1",
            "clock_basis": "dispatcher_token_observation_perf_counter",
            "includes_gpu_host_delivery_and_scheduling": True,
            "server_ttft_ms": (first - request_started_at) * 1000 if complete else None,
            "server_first_token_event_ms": (first - request_started_at) * 1000 if events and ordered else None,
            "server_decode_ms": decode_ms,
            "server_tpot_ms": decode_ms / (output_tokens - 1) if complete and output_tokens > 1 else None,
            "server_queue_ms": (self.scheduled_at - self.queued_at) * 1000
                if self.scheduled_at is not None and self.queued_at is not None and self.scheduled_at >= self.queued_at else None,
            "server_prefill_ms": (first - self.scheduled_at) * 1000
                if complete and self.scheduled_at is not None and first >= self.scheduled_at else None,
            "server_queue_to_first_token_ms": (first - self.queued_at) * 1000
                if complete and self.queued_at is not None and first >= self.queued_at else None,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "dispatcher_prepare_ms": (self.queued_at - self.submitted_at) * 1000 if self.queued_at is not None else None,
            "dispatcher_admission_wait_ms": (self.admitted_at - self.queued_at) * 1000
                if self.admitted_at is not None and self.queued_at is not None else None,
            "observed_output_tokens": observed,
            "token_event_count": len(events),
            "coalesced_token_events": coalesced,
            "token_timing_complete": complete,
            "token_itl_ms": gaps if complete else None,
            "token_event_gaps_ms": gaps,
        }
