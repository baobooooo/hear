"""Wire the sampler and the per-request engine feedback into run_timeline.py.

Adds:
  * --gpu N  : which card nvidia-smi should watch (default 0)
  * "samples": the time series from harness/sampler.py in the run JSON
  * per-turn "kvp": whatever the engine returned in kv_transfer_params
    (protection fired? how long it waited? L1 vs L2 tokens)
"""
from pathlib import Path

p = Path("harness/run_timeline.py")
s = p.read_text()
if "sampler" in s:
    print("already wired")
    raise SystemExit

# import
old = "from oracle import CacheOracle"
assert s.count(old) == 1
s = s.replace(old, "from oracle import CacheOracle\nfrom sampler import Sampler")

# capture kv_transfer_params on each completion
old = """            t.cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
"""
assert s.count(old) == 1
s = s.replace(old, old + """            t.kvp = r.get("kv_transfer_params") or None
""")

old = """        self.pred_gpu = self.pred_cpu = 0
"""
assert s.count(old) == 1
s = s.replace(old, old + """        self.kvp = None
""")

# row(): carry it through
old = """    def row(self):
        d = {k: getattr(self, k) for k in ("""
assert s.count(old) == 1
s = s.replace(old, """    def row(self):
        extra = {"kvp": self.kvp} if self.kvp else {}
        d = {k: getattr(self, k) for k in (""")
i = s.index("    def row(self):")
j = s.index("return d", i)
assert s[j:j + 8] == "return d"
s = s[:j] + "d.update(extra)\n        return d" + s[j + 8:]

# cli
old = '    ap.add_argument("--capacity", type=int, default=518080)\n'
assert s.count(old) == 1
s = s.replace(old, old + '    ap.add_argument("--gpu", type=int, default=0,\n'
                         '                    help="card index for nvidia-smi sampling")\n'
                         '    ap.add_argument("--sample-s", type=float, default=0.5)\n')

# start the sampler right after the caches are reset, stop it after the run
old = "    evq: queue.Queue = queue.Queue()\n"
assert s.count(old) == 1
s = s.replace(old, "    sampler = Sampler(args.base, args.gpu, lambda: now() - t_zero[0], args.sample_s)\n" + old)

old = "    final = graph.invoke("
assert s.count(old) == 1
s = s.replace(old, "    sampler.start()\n    " + old)

old = "    R = final[\"records\"]\n"
assert s.count(old) == 1
s = s.replace(old, "    sampler.stop()\n" + old)

old = '        "prefetches": pf_log, "records": R,\n'
assert s.count(old) == 1
s = s.replace(old, '        "prefetches": pf_log, "samples": sampler.samples, "records": R,\n')

p.write_text(s)
print("instrumentation wired")
