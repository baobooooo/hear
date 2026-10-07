"""Expose --max-num-batched-tokens as MAX_BATCHED in the launcher (0 = vLLM default)."""
from pathlib import Path

p = Path("scripts/launch_vllm.sh")
s = p.read_text()
if "MAX_BATCHED" in s:
    print("already present")
    raise SystemExit
old = "MAX_SEQS=${MAX_SEQS:-40}\n"
assert s.count(old) == 1
s = s.replace(old, old + "MAX_BATCHED=${MAX_BATCHED:-0}\n")
old = "extra=()\n"
assert s.count(old) == 1
s = s.replace(old, old + '(( MAX_BATCHED > 0 )) && extra+=(--max-num-batched-tokens "$MAX_BATCHED")\n')
p.write_text(s)
print("MAX_BATCHED knob added")
