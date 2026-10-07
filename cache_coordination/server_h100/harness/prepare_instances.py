"""Build token-id conversation skeletons with a controlled spread of prefix
lengths and turn counts.

Each SCBench kv row is a flat JSON map of uuid -> uuid plus "what is the value
for this key" questions. We keep a window of that map to land on a chosen
prompt length, and ask about keys drawn from the kept window, so every question
stays answerable no matter how short we cut the context. The question wording
is SCBench's own.

A row has ~1250 entries and a 20k-token prompt uses ~290 of them, so one row
yields several *disjoint* windows. Each window is a different set of uuid pairs
and therefore a different prefix, which is how we get a few hundred distinct
conversations out of 100 rows without new data.

Every turn's prompt is built at the token-id level so that turn k is a strict
token-wise extension of turn k-1: that is what makes the prefix cache reusable
across turns and what lets the replay arms send byte-identical prompts.

    turn 1 prompt = SYS + USER_OPEN + ctx + Q1 + TURN_END + ASSIST_OPEN + NOTHINK
    turn k prompt = turn k-1 prompt + A(k-1) + TURN_END + USER_OPEN + Qk
                    + TURN_END + ASSIST_OPEN + NOTHINK
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer

SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
USER_OPEN = "<|im_start|>user\n"
TURN_END = "<|im_end|>\n"
ASSIST_OPEN = "<|im_start|>assistant\n"
NOTHINK = "<think>\n\n</think>\n\n"   # Qwen3 non-thinking convention

QUESTION = '\nKey: "%s"\nThe value associated with the specified key is: '
ENTRY_RE = re.compile(r'"([0-9a-f-]{36})": "([0-9a-f-]{36})"')


def parse_ctx(ctx):
    head, _, body = ctx.partition("{")
    return head, ENTRY_RE.findall(body)


def render_ctx(head, entries):
    return head + "{" + ", ".join('"%s": "%s"' % kv for kv in entries) + "}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/scbench_kv_mid.jsonl")
    ap.add_argument("--model", default="/data/huggingface/Qwen3-8B")
    ap.add_argument("--out", default="data/instances_het.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--windows", type=int, default=1,
                    help="disjoint context windows to carve from each row")
    ap.add_argument("--targets", default="8000,14000,20000,26000,32000,38000")
    ap.add_argument("--turns-spread", default="3,4,5,6")
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--answer-budget", type=int, default=96)
    args = ap.parse_args()

    targets = [int(x) for x in args.targets.split(",")]
    turn_spread = [int(x) for x in args.turns_spread.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    sys_ids, user_open = enc(SYS), enc(USER_OPEN)
    turn_end, assist_open, nothink = enc(TURN_END), enc(ASSIST_OPEN), enc(NOTHINK)
    fixed = len(sys_ids) + len(user_open) + len(turn_end) + len(assist_open) + len(nothink)
    q_len = len(enc(QUESTION % ("0" * 36)))

    rows = [json.loads(l) for l in Path(args.src).read_text().splitlines()]
    head0, ent0 = parse_ctx(rows[0]["prompts"][0])
    per_entry = len(enc(render_ctx(head0, ent0[:200]))) / 200.0
    print("loaded %d rows, %d entries/context, ~%.1f tokens/entry"
          % (len(rows), len(ent0), per_entry))

    # candidate (row, window) slots, interleaved so consecutive instances come
    # from different rows
    slots = [(k, w) for w in range(args.windows) for k in range(len(rows))]

    picked, skipped = [], 0
    for idx, (k, w) in enumerate(slots):
        if len(picked) >= args.n:
            break
        row = rows[k]
        target = targets[idx % len(targets)]
        turns = turn_spread[idx % len(turn_spread)]
        head_txt, entries = parse_ctx(row["prompts"][0])

        keep = max(turns, int((target - fixed - q_len) / per_entry))
        lo, hi = w * keep, (w + 1) * keep
        if hi > len(entries):
            skipped += 1
            continue
        kept = entries[lo:hi]

        step = max(1, keep // turns)
        qa = [kept[min(keep - 1, i * step + step // 2)] for i in range(turns)]
        ctx_ids = enc(render_ctx(head_txt, kept))
        q_ids = [enc(QUESTION % key) for key, _ in qa]
        head = (sys_ids + user_open + ctx_ids + q_ids[0] + turn_end
                + assist_open + nothink)
        suffixes = [turn_end + user_open + q + turn_end + assist_open + nothink
                    for q in q_ids[1:]]
        final = len(head) + sum(len(s) for s in suffixes) + args.answer_budget * turns
        if final > args.max_len:
            skipped += 1
            continue

        picked.append({
            "id": "%s.w%d@%dk-t%d" % (row["id"], w, target // 1000, turns),
            "head_ids": head, "turn_suffix_ids": suffixes,
            "ground_truth": [v for _, v in qa],
            "turn1_len": len(head), "final_len_worst_case": final,
            "turns": turns, "target": target,
        })

    plens = [c["turn1_len"] for c in picked]
    print("\nselected %d instances (%d slots skipped)" % (len(picked), skipped))
    print("  turn-1 prompt : min=%d med=%d mean=%d max=%d"
          % (min(plens), int(statistics.median(plens)), sum(plens) // len(plens), max(plens)))
    from collections import Counter
    print("  by target     : %s" % dict(sorted(Counter(c["target"] for c in picked).items())))
    print("  by turns      : %s" % dict(sorted(Counter(c["turns"] for c in picked).items())))
    print("  total requests: %d" % sum(c["turns"] for c in picked))
    print("  working set   : %d tokens = %.1f GiB"
          % (sum(plens), sum(plens) * 144 / 1024 / 1024))

    Path(args.out).write_text(json.dumps({
        "source": args.src, "model": args.model, "turns": max(turn_spread),
        "answer_budget": args.answer_budget, "instances": picked,
    }))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
