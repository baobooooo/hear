"""Harness: send a per-turn return-probability hint (`p_return`) to the engine.

--p-return-table '{"1":0.31,"2":0.42,"3":0.55,"4":0.56,"5":0.81}'  maps the
session depth of the turn being sent (1 = first turn) to P(user comes back for
another turn). Calibrated on the Mooncake trace's first 10 minutes (before the
replayed window). When given, every request (baseline or kvprotect) carries
vllm_xargs.p_return; the KVP_RETAIN engine patch uses it for L2 retention.
"""
from pathlib import Path

p = Path("harness/run_timeline.py")
s = p.read_text()
if "p_return" in s:
    print("already patched")
    raise SystemExit

old = '    ap.add_argument("--protect-s", type=float, default=60.0,\n'
assert s.count(old) == 1
s = s.replace(old, '    ap.add_argument("--p-return-table", default=None,\n'
                   '                    help="JSON {depth: P(next turn)}; sends vllm_xargs.p_return with every request")\n' + old)

# send(): merge the hint into xargs
old = '''    def send(tid, prio, xargs=None):
        t = turns[tid]
        t.priority = prio
'''
assert s.count(old) == 1
s = s.replace(old, '''    def send(tid, prio, xargs=None):
        t = turns[tid]
        t.priority = prio
        if cfg.get("p_return_table"):
            tab = cfg["p_return_table"]
            pr = tab.get(str(t.r + 1), tab[max(tab, key=int)])
            xargs = {**(xargs or {}), "p_return": pr}
''')

old = '''                   "jump_s": args.jump_s, "protect_s": args.protect_s})'''
assert s.count(old) == 1
s = s.replace(old, '''                   "jump_s": args.jump_s, "protect_s": args.protect_s,
                   "p_return_table": json.loads(args.p_return_table) if args.p_return_table else None})''')

old = '        "capacity": args.capacity, "jump_s": args.jump_s, "protect_s": args.protect_s,\n'
assert s.count(old) == 1
s = s.replace(old, old + '        "p_return_table": args.p_return_table,\n')
p.write_text(s)
print("p_return hint wired")
