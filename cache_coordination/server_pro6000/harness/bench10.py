"""Cold-batch reference point: N requests with distinct prefixes, L1 only.

Shuffled so no two consecutive requests share a prefix, and the prefix cache is
reset first, so every request must prefill from scratch. This is the "nothing
hits" floor that every cache-aware number should be compared against.

Reports the batch makespan for all N dispatched at once, and for the same N run
one at a time, plus the achieved prefill throughput.
"""

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def reset(base):
    root = base.rstrip("/").rsplit("/v1", 1)[0]
    urllib.request.urlopen(urllib.request.Request(
        root + "/reset_prefix_cache", data=b"",
        headers={"Content-Type": "application/json"}), timeout=60).read()
    time.sleep(2.0)


def post(url, model, prompt, n_out):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        r = json.loads(resp.read())
    dt = time.perf_counter() - t0
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return dt, u["prompt_tokens"], cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/instances_320.json")
    ap.add_argument("--traj", default="trajectories/trajectories_320.json")
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-out", type=int, default=0,
                    help="force this many output tokens; 0 = use recorded length")
    ap.add_argument("--also-sequential", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())["instances"]
    traj = json.loads(Path(args.traj).read_text())["trajectories"]
    url = args.base.rstrip("/") + "/completions"

    rng = random.Random(args.seed)
    picks = rng.sample(spec, args.n)          # distinct conversations
    rng.shuffle(picks)                        # shuffled dispatch order
    jobs = []
    for inst in picks:
        out = traj[inst["id"]]["turns"][0]["output_ids"]
        jobs.append((inst["id"], inst["head_ids"],
                     args.n_out or len(out)))

    lens = [len(p) for _, p, _ in jobs]
    outs = [o for _, _, o in jobs]
    print("%d requests, distinct prefixes, shuffled" % args.n)
    print("  prompt tokens: %s" % lens)
    print("  total prompt  : %d tokens   output: %d tokens" % (sum(lens), sum(outs)))

    print("\n--- concurrent (all %d dispatched at once) ---" % args.n)
    reset(args.base)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.n) as ex:
        res = list(ex.map(lambda j: post(url, args.model, j[1], j[2]), jobs))
    wall = time.perf_counter() - t0
    cached = sum(r[2] for r in res)
    lat = sorted(r[0] for r in res)
    print("  makespan      : %.2f s" % wall)
    print("  cache hit     : %d tokens (%.2f%%)  <- should be ~0"
          % (cached, 100 * cached / sum(lens)))
    print("  per-request   : min %.2fs med %.2fs max %.2fs"
          % (lat[0], lat[len(lat) // 2], lat[-1]))
    print("  prompt tput   : %.0f tok/s" % (sum(lens) / wall))

    if args.also_sequential:
        print("\n--- sequential (one at a time) ---")
        reset(args.base)
        t0 = time.perf_counter()
        seq = [post(url, args.model, j[1], j[2]) for j in jobs]
        swall = time.perf_counter() - t0
        print("  makespan      : %.2f s" % swall)
        print("  cache hit     : %d tokens" % sum(r[2] for r in seq))
        print("  prompt tput   : %.0f tok/s" % (sum(lens) / swall))
        print("\n  concurrent is %.2fx faster than sequential" % (swall / wall))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
