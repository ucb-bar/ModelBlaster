cd /tmp/krisd/a2/f48/work; python3 - <<'PY'
import json, glob
t = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
for f in glob.glob("r0*/run/*/*/result.json") + glob.glob("r0*/variants/*/run/*/result.json") + glob.glob("r0*/**/result.json", recursive=True):
    pass
seen = set()
for f in glob.glob("**/result.json", recursive=True):
    if f in seen: continue
    seen.add(f)
    r = json.load(open(f)); u = r.get("llm_usage") or {}
    t["prompt_tokens"] += u.get("prompt_tokens", 0); t["completion_tokens"] += u.get("completion_tokens", 0); t["calls"] += r.get("llm_calls") or 0
print(len(seen), t, "usd(in*5+out*25)", round((t["prompt_tokens"] * 5 + t["completion_tokens"] * 25) / 1e6, 2))
PY
date; tail -1 /tmp/krisd/farm/llm_ledger.jsonl | cut -c1-60
