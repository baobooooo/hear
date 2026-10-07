"""Pass 0: record one trajectory per SCBench instance at the token-id level.

The replay passes must send byte-identical prompts in both arms, but turn k's
prompt normally depends on the model's turn k-1 output, and vLLM is not
batch-invariant -- different scheduling orders can change the generated tokens.
So we record the answers once here and teacher-force them during replay.

Output: trajectories.json
    {instance_id: {"turns": [{"prompt_len": int, "output_ids": [...]}, ...]}}
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IM_END = 151645  # <|im_end|> for Qwen3


def complete(url: str, model: str, prompt: list[int], max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop_token_ids": [IM_END],
        "return_token_ids": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def token_ids_of(choice: dict) -> list[int]:
    for key in ("token_ids", "completion_token_ids", "output_token_ids"):
        if choice.get(key):
            return list(choice[key])
    raise KeyError("no output token ids in choice: %s" % sorted(choice))


def record_one(url: str, model: str, inst: dict, budget: int) -> dict:
    ids = list(inst["head_ids"])
    turns = []
    for t in range(len(inst["turn_suffix_ids"]) + 1):
        r = complete(url, model, ids, budget)
        choice = r["choices"][0]
        out = token_ids_of(choice)
        turns.append({
            "prompt_len": len(ids),
            "output_ids": out,
            "text": choice["text"][:120],
        })
        if t < len(inst["turn_suffix_ids"]):
            ids = ids + out + inst["turn_suffix_ids"][t]
    return {"id": inst["id"], "turns": turns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/instances.json")
    ap.add_argument("--out", default="trajectories/trajectories.json")
    ap.add_argument("--base", default="http://127.0.0.1:19081/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--conc", type=int, default=6)
    args = ap.parse_args()

    spec = json.loads(Path(args.instances).read_text())
    insts = spec["instances"]
    budget = spec["answer_budget"]
    url = args.base.rstrip("/") + "/completions"
    print("recording %d instances x %d turns, budget=%d tokens/turn"
          % (len(insts), spec["turns"], budget))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.conc) as ex:
        recs = list(ex.map(lambda i: record_one(url, args.model, i, budget), insts))
    wall = time.perf_counter() - t0

    out_lens = [len(t["output_ids"]) for r in recs for t in r["turns"]]
    print("done in %.1fs" % wall)
    print("output tokens per turn: min=%d mean=%.1f max=%d"
          % (min(out_lens), sum(out_lens) / len(out_lens), max(out_lens)))
    print("sample answer: %r" % recs[0]["turns"][0]["text"])
    truncated = sum(1 for n in out_lens if n >= budget)
    if truncated:
        print("NOTE: %d/%d turns hit the %d-token budget (no natural stop)"
              % (truncated, len(out_lens), budget))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "instances": args.instances,
        "model": args.model,
        "record_wall_s": wall,
        "trajectories": {r["id"]: r for r in recs},
    }))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
