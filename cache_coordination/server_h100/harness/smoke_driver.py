"""Smoke-test the three engine->harness signals the KV-aware harness relies on.

1. /v1/completions accepts a raw token-id prompt (so replay can be byte-exact).
2. usage.prompt_tokens_details.cached_tokens reports the prefix hit.
3. Driving more prefix bytes than L1 holds pushes blocks into the CPU (L2) tier,
   which should show up on the KV event stream as medium=CPU.

Run the event probe alongside this to see the events themselves.
"""

import argparse
import json
import random
import time
import urllib.request

VOCAB_SAFE = 150000


def make_prefix(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, VOCAB_SAFE) for _ in range(n)]


def complete(url: str, model: str, prompt: list[int], n_out: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": n_out,
            "min_tokens": n_out,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    dt = time.perf_counter() - t0
    usage = payload.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    return {
        "latency_s": dt,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "text": payload["choices"][0]["text"][:40],
    }


def show(label: str, r: dict) -> None:
    print(
        "%-28s latency=%6.2fs prompt=%6s cached=%6s out=%s"
        % (label, r["latency_s"], r["prompt_tokens"], r["cached_tokens"],
           r["completion_tokens"]),
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--prefix-len", type=int, default=20000)
    ap.add_argument("--n-fill", type=int, default=6,
                    help="distinct prefixes used to overflow L1")
    ap.add_argument("--out", type=int, default=8)
    args = ap.parse_args()
    url = args.base.rstrip("/") + "/completions"

    base = make_prefix(0, args.prefix_len)

    print("== signal 1+2: token-id prompt and cached_tokens ==")
    show("cold  prefix#0", complete(url, args.model, base, args.out))
    show("warm  prefix#0 (same)", complete(url, args.model, base, args.out))
    show("warm  prefix#0 + 200 new",
         complete(url, args.model, base + make_prefix(99, 200), args.out))

    print("\n== signal 3: overflow L1 with %d x %d tokens =="
          % (args.n_fill, args.prefix_len))
    for i in range(1, args.n_fill + 1):
        show("fill  prefix#%d" % i,
             complete(url, args.model, make_prefix(i, args.prefix_len), args.out))

    print("\n== after eviction: does prefix#0 still hit (via L2)? ==")
    show("revisit prefix#0", complete(url, args.model, base, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
