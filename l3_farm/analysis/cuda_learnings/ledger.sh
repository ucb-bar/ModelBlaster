L=/tmp/krisd/farm/llm_ledger.jsonl
if [ -f $L ]; then python3 - $L <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
t = {k: sum((r.get(k) or 0) for r in rows) for k in ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_write_input_tokens", "est_usd")}
cost = (t["prompt_tokens"] * 5 + t["completion_tokens"] * 25 + t["cache_write_input_tokens"] * 6.25 + t["cache_read_input_tokens"] * 0.5) / 1e6
print(json.dumps({"rows": len(rows), "calls": sum(r.get("calls") or 1 for r in rows), **t, "usd_opus": round(cost, 2),
                  "first": rows[0].get("ts") if rows else None, "last": rows[-1].get("ts") if rows else None}))
PY
else echo '{"rows": 0}'; fi
