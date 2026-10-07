"""One-off edits on the server: launcher env passthrough + kvprotect group."""
from pathlib import Path

# 1) launcher: pass the patch directory and switches to the engine
p = Path("scripts/launch_vllm.sh")
s = p.read_text()
old = "  VLLM_SERVER_DEV_MODE=1 \\\n"
new = old + ('  PYTHONPATH="${KVP_DIR:-}" KVP_PROTECT="${KVP_PROTECT:-0}" '
             'KVP_KEEPALIVE="${KVP_KEEPALIVE:-0}" KVP_KEEPALIVE_S="${KVP_KEEPALIVE_S:-0.5}" \\\n')
if "KVP_PROTECT" not in s:
    assert s.count(old) == 1
    s = s.replace(old, new)
    p.write_text(s)

# 2) harness: vllm_xargs on post(), kvprotect group
p = Path("harness/run_timeline.py")
s = p.read_text()
if "kvprotect" not in s:
    old = '''def post(url, model, prompt, n_out, priority=0):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }).encode()
'''
    new = '''def post(url, model, prompt, n_out, priority=0, xargs=None):
    body = {
        "model": model, "prompt": prompt,
        "max_tokens": n_out, "min_tokens": n_out,
        "ignore_eos": True, "temperature": 0.0, "stream": False,
        "priority": priority,
    }
    if xargs:
        body["vllm_xargs"] = xargs
    body = json.dumps(body).encode()
'''
    assert old in s
    s = s.replace(old, new)

    old = '''    def send(tid, prio):
        t = turns[tid]
        t.priority = prio
'''
    new = '''    def send(tid, prio, xargs=None):
        t = turns[tid]
        t.priority = prio
'''
    assert old in s
    s = s.replace(old, new)
    old = "            r = post(url, model, t.prompt, t.n_out, prio)\n"
    assert s.count(old) == 1
    s = s.replace(old, "            r = post(url, model, t.prompt, t.n_out, prio, xargs)\n")

    old = '''        if group in ("kvaware_fair", "kvaware_slo"):'''
    new = '''        if group == "kvprotect":
            # everything goes to the engine at once; the engine protects a turn
            # once it has waited protect_s there (deadline promotion patch)
            for tid in pending:
                t = turns[tid]
                gpu, cpu, _ = oracle.match(t.prompt)
                prio = -((gpu + cpu) // 16) if (gpu + cpu) >= 0.5 * len(t.prompt) else 0
                send(tid, prio, {"protect_s": cfg["protect_s"]})
            return {"pending": [], "events": state["events"] + 1}

        if group in ("kvaware_fair", "kvaware_slo"):'''
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = '"kvaware_slo", "kvaware_ft"])'
    assert s.count(old) == 1
    s = s.replace(old, '"kvaware_slo", "kvaware_ft", "kvprotect"])')
    old = '    ap.add_argument("--jump-s", type=float, default=60.0,'
    assert s.count(old) == 1
    s = s.replace(old, '    ap.add_argument("--protect-s", type=float, default=60.0,\n'
                       '                    help="kvprotect: engine-side protection after this many seconds of waiting")\n' + old)
    old = '                   "jump_s": args.jump_s})'
    assert s.count(old) == 1
    s = s.replace(old, '                   "jump_s": args.jump_s, "protect_s": args.protect_s})')
    old = '        "capacity": args.capacity, "jump_s": args.jump_s,'
    assert s.count(old) == 1
    s = s.replace(old, '        "capacity": args.capacity, "jump_s": args.jump_s, "protect_s": args.protect_s,')
    p.write_text(s)
print("edits applied")
