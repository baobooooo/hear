"""Measure the prefill and decode rates the speedup model needs.

Prefill: cold 20k-token prompt with 1 output token.
Decode : the same prompt warm (cached_tokens ~= prompt), so the measured time is
         almost entirely decode. Run at concurrency 1 and 8.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:19081/v1/completions"


def mk(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def call(prompt: list[int], n_out: int) -> tuple[float, int, int]:
    body = json.dumps({
        "model": "Qwen3-8B",
        "prompt": prompt,
        "max_tokens": n_out,
        "min_tokens": n_out,
        "ignore_eos": True,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.perf_counter() - t0
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return dt, u["completion_tokens"], cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=20000)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9000)
    args = ap.parse_args()
    P, D, C = args.prefix, args.decode, args.conc

    print("== prefill (cold, 1 output token) ==")
    cold = []
    for i in range(3):
        dt, _, cached = call(mk(args.seed0 + i, P), 1)
        cold.append(dt)
        print("  cold %d: %.3fs cached=%d -> %.0f tok/s" % (i, dt, cached, P / dt))
    t_pref = statistics.median(cold)
    print("  median prefill: %.3fs -> %.0f tok/s" % (t_pref, P / t_pref))

    print("\n== decode, concurrency 1 (prefix warm) ==")
    warm = mk(args.seed0, P)
    call(warm, 1)  # make sure it is hot
    dt, n, cached = call(warm, D)
    print("  %.3fs for %d tokens (cached=%d) -> %.1f tok/s/req"
          % (dt, n, cached, n / dt))
    single = n / dt

    print("\n== decode, concurrency %d (all prefixes warm) ==" % C)
    prompts = [mk(args.seed0 + 100 + i, P) for i in range(C)]
    with ThreadPoolExecutor(C) as ex:
        list(ex.map(lambda p: call(p, 1), prompts))  # warm them
    t0 = time.perf_counter()
    with ThreadPoolExecutor(C) as ex:
        res = list(ex.map(lambda p: call(p, D), prompts))
    wall = time.perf_counter() - t0
    total_out = sum(r[1] for r in res)
    hit = sum(1 for r in res if r[2] >= P - 32)
    print("  wall %.3fs, %d output tokens, %d/%d prefixes hit cache"
          % (wall, total_out, hit, C))
    print("  aggregate decode: %.0f tok/s   per-request: %.1f tok/s"
          % (total_out / wall, total_out / wall / C))

    print("\n== summary for the speedup model ==")
    print("  prefill        : %.0f tok/s" % (P / t_pref))
    print("  decode  bs=1   : %.1f tok/s" % single)
    print("  decode  bs=%-3d : %.0f tok/s aggregate" % (C, total_out / wall))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
