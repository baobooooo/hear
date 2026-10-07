"""Make every run start from an empty L1 *and* L2.

/reset_prefix_cache without ?reset_external=true only clears the GPU prefix
cache; the CPU offload tier (L2) kept the previous run's KV, so the next run's
first turns (same instances, same seed) hit it. This edit resets both tiers,
then verifies with a probe: after reset, one first-turn prompt must come back
with (almost) nothing cached; the probe's own KV is removed by a second reset.
"""
from pathlib import Path

HELPER = '''

def reset_all_caches(root, base, model, probe_prompt):
    """Clear L1 + L2, prove it with a probe, clear again. Raises if not clean."""
    def reset():
        for _ in range(30):
            with urllib.request.urlopen(urllib.request.Request(
                    root + "/reset_prefix_cache?reset_external=true", data=b"",
                    headers={"Content-Type": "application/json"}), timeout=60) as r:
                if r.status == 200:
                    return
            time.sleep(1.0)
        raise RuntimeError("reset_prefix_cache did not succeed")
    reset()
    time.sleep(2.0)
    r = post(base.rstrip("/") + "/completions", model, probe_prompt, 1)
    cached = ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if cached > 256:
        raise RuntimeError("cache not empty after reset: probe hit %d tokens" % cached)
    reset()
    time.sleep(2.0)
    print("cache reset verified (L1+L2): probe cached %d tokens" % cached, flush=True)
'''


def patch(path, old_call, probe_expr, url_expr, model_expr):
    p = Path(path)
    s = p.read_text()
    if "reset_all_caches" in s:
        print(path, "already patched")
        return
    assert s.count(old_call) == 1, path
    s = s.replace(old_call, "    reset_all_caches(root, %s, %s, %s)\n" % (url_expr, model_expr, probe_expr))
    anchor = "\n\nclass Conv:" if "\nclass Conv:" in s else None
    if anchor is None:
        # put helper right after post()
        i = s.index("\ndef post(")
        j = s.index("\n\n\n", i)
        s = s[:j] + HELPER + s[j:]
    else:
        s = s.replace(anchor, HELPER + anchor, 1)
    p.write_text(s)
    print(path, "patched")


OLD = '''    urllib.request.urlopen(urllib.request.Request(
        root + "/reset_prefix_cache", data=b"",
        headers={"Content-Type": "application/json"}), timeout=60).read()
'''
patch("harness/run_timeline.py", OLD, "convs[0].prompts[0]", "args.base", "args.model")
