"""Teach run_timeline.py to replay a Mooncake workload (kind == "mooncake").

Prompt tokens are synthesised deterministically per 512-token hash block, so
two requests that share a hash id share identical tokens -- the trace's prefix
structure is reproduced exactly (including the partial last block that is
re-hashed on the next turn). Output length is the trace's (forced with
min_tokens = max_tokens, ignore_eos).
"""
from pathlib import Path

p = Path("harness/run_timeline.py")
s = p.read_text()
if "load_mooncake" in s:
    print("already patched")
    raise SystemExit

helper = '''

MC_BLOCK = 512
_mc_blocks: dict = {}


def mc_block_tokens(h):
    """Deterministic token block for a Mooncake hash id (plain vocab range, no specials)."""
    b = _mc_blocks.get(h)
    if b is None:
        rng = random.Random("mooncake:%d" % h)
        b = [rng.randrange(1000, 120000) for _ in range(MC_BLOCK)]
        _mc_blocks[h] = b
    return b


class MConv:
    """A replayed Mooncake session: one synthesized prompt per turn."""

    def __init__(self, k, se):
        self.id = "mc%d" % k
        self.prompts, self.out_lens = [], []
        for t in se["turns"]:
            toks = []
            for h in t["hash_ids"]:
                toks.extend(mc_block_tokens(h))
            self.prompts.append(toks[:t["input_length"]])
            self.out_lens.append(max(1, int(t["output_length"])))
        self.history = list(self.prompts)


def load_mooncake(wl):
    convs, arrivals, thinks, n_turns = [], [], [], []
    for k, se in enumerate(wl["sessions"]):
        c = MConv(k, se)
        convs.append(c)
        arrivals.append(float(se["start"]))
        thinks.append(list(se["thinks"]) + [0.0])
        n_turns.append(len(c.prompts))
    return convs, arrivals, thinks, n_turns

'''
old = "\n\ndef load_workload(wl, spec, traj, seed):"
assert s.count(old) == 1
s = s.replace(old, helper + "\ndef load_workload(wl, spec, traj, seed):")

old = '''        convs, arrivals, thinks, n_turns = load_workload(wl, spec, traj, args.seed)
'''
assert s.count(old) == 1
s = s.replace(old, '''        if wl["meta"].get("kind") == "mooncake":
            convs, arrivals, thinks, n_turns = load_mooncake(wl)
        else:
            convs, arrivals, thinks, n_turns = load_workload(wl, spec, traj, args.seed)
''')
p.write_text(s)
print("mooncake replay wired")
