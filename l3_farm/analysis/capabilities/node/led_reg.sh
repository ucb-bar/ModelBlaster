export LO=2026-10-25T17:17 HI=2026-10-25T17:39
L=/tmp/krisd/farm/llm_ledger.jsonl
[ -f $L ] && python3 - $L <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
import os
lo, hi = os.environ.get("LO", "2026-10-25T16:48"), os.environ.get("HI", "9999")
rows = [r for r in rows if lo <= str(r.get("ts", "")) < hi]   # node-local time window
t = {k: sum((r.get(k) or 0) for r in rows) for k in ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_write_input_tokens")}
cost = (t["prompt_tokens"] * 5 + t["completion_tokens"] * 25 + t["cache_write_input_tokens"] * 6.25 + t["cache_read_input_tokens"] * 0.5) / 1e6
print(json.dumps({"rows": len(rows), **t, "usd_opus": round(cost, 2), "first": rows[0].get("ts") if rows else None, "last": rows[-1].get("ts") if rows else None}))
PY
tail -1 $L | cut -c1-300
