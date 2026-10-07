"""Teach run_timeline.py to replay a trace-calibrated workload file.

--workload W.json  (from build_workload.py) overrides --n / arrival / think
sampling: every session gets its own SCBench instance (never reused, so no
accidental cross-user prefix sharing), its trace turn count (capped by the
instance) and its trace think-time sequence.
"""
from pathlib import Path

p = Path("harness/run_timeline.py")
s = p.read_text()
if "--workload" in s:
    print("already patched")
    raise SystemExit

# 1) loader helper, placed before main()
helper = '''

def load_workload(wl, spec, traj, seed):
    """sessions -> (convs, arrivals, thinks, n_turns); one unique instance per session."""
    sess = wl["sessions"]
    rng = random.Random("%d:workload" % seed)
    pool = sorted(spec, key=lambda i: i["id"])
    rng.shuffle(pool)
    if len(sess) > len(pool):
        raise SystemExit("workload has %d sessions but only %d unique instances" % (len(sess), len(pool)))
    convs, arrivals, thinks, n_turns = [], [], [], []
    for inst, se in zip(pool, sess):
        c = Conv(inst, traj[inst["id"]])
        n = min(se["n"], len(c.prompts))
        z = list(se["thinks"][:n - 1]) + [0.0]
        convs.append(c)
        arrivals.append(float(se["start"]))
        thinks.append(z)
        n_turns.append(n)
    return convs, arrivals, thinks, n_turns

'''
old = "\n\ndef main() -> int:"
assert s.count(old) == 1
s = s.replace(old, helper + "\ndef main() -> int:")

# 2) CLI flag
old = '    ap.add_argument("--n", type=int, default=100)\n'
assert s.count(old) == 1
s = s.replace(old, old + '    ap.add_argument("--workload", default=None,\n'
                         '                    help="trace-calibrated sessions (build_workload.py); overrides --n/arrival/think")\n')

# 3) after the synthetic sampling, override when a workload is given
old = "    total = sum(len(c.prompts) for c in convs)\n"
assert s.count(old) == 1
s = s.replace(old, old + '''    n_turns = [len(c.prompts) for c in convs]
    workload_meta = None
    if args.workload:
        wl = json.loads(Path(args.workload).read_text())
        workload_meta = wl["meta"]
        convs, arrivals, thinks, n_turns = load_workload(wl, spec, traj, args.seed)
        total = sum(n_turns)
''')

# 4) the user thread respects the per-session turn count
old = "        for r in range(len(c.prompts)):\n"
assert s.count(old) == 1
s = s.replace(old, "        for r in range(n_turns[i]):\n")
old = "            if r + 1 < len(c.prompts):\n                time.sleep(thinks[i][r])\n"
assert s.count(old) == 1
s = s.replace(old, "            if r + 1 < n_turns[i]:\n                time.sleep(thinks[i][r])\n")

# 5) sampling too many users with --n when a workload is used: pick at least as many instances
old = "    picked = rng.sample(spec, args.n)\n"
assert s.count(old) == 1
s = s.replace(old, "    picked = rng.sample(spec, min(args.n, len(spec)))\n")

# 6) record the workload in the output
old = '        "arrival_dist": args.arrival_dist,\n'
assert s.count(old) == 1
s = s.replace(old, old + '        "workload": workload_meta, "workload_file": args.workload,\n')

p.write_text(s)
print("workload replay wired")
