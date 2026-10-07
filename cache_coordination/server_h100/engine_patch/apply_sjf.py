"""Harness: optional shortest-prefill-first ordering for cold requests (kvprotect group).

--cold-sjf  : cold requests get priority = uncached prompt length / 256 (shorter
              first) instead of 0; hot requests keep priority = -(cached blocks).
              The engine-side protection (W) still bounds how long anything can
              be overtaken, so the tail stays protected while the mean drops.
"""
from pathlib import Path

p = Path("harness/run_timeline.py")
s = p.read_text()
if "cold_sjf" in s:
    print("already patched")
    raise SystemExit

old = '''                prio = -((gpu + cpu) // 16) if (gpu + cpu) >= 0.5 * len(t.prompt) else 0
                send(tid, prio, {"protect_s": cfg["protect_s"]})'''
assert s.count(old) == 1
s = s.replace(old, '''                if (gpu + cpu) >= 0.5 * len(t.prompt):
                    prio = -((gpu + cpu) // 16)
                elif cfg.get("cold_sjf"):
                    prio = max(1, (len(t.prompt) - gpu - cpu) // 256)     # shorter prefill first
                else:
                    prio = 0
                send(tid, prio, {"protect_s": cfg["protect_s"]})''')

old = '    ap.add_argument("--p-return-table", default=None,\n'
assert s.count(old) == 1
s = s.replace(old, '    ap.add_argument("--cold-sjf", action="store_true",\n'
                   '                    help="kvprotect: cold requests ordered shortest-prefill-first (bounded by protect_s)")\n' + old)

old = '''                   "p_return_table": json.loads(args.p_return_table) if args.p_return_table else None})'''
assert s.count(old) == 1
s = s.replace(old, '''                   "p_return_table": json.loads(args.p_return_table) if args.p_return_table else None,
                   "cold_sjf": args.cold_sjf})''')
old = '        "p_return_table": args.p_return_table,\n'
assert s.count(old) == 1
s = s.replace(old, old + '        "cold_sjf": args.cold_sjf,\n')
p.write_text(s)
print("cold-sjf wired")
