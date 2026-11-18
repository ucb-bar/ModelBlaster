import json, sys
# the node's relay ledger (my nodes only ran my work): every row; prompt tokens at the full input price (upper bound)
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][int(sys.argv[2]) if len(sys.argv) > 2 else 0:]   # skip rows from before
t = {k: sum((r.get(k) or 0) for r in rows) for k in ("calls", "prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_write_input_tokens")}
cost = (t["prompt_tokens"] * 5 + t["completion_tokens"] * 25 + t["cache_write_input_tokens"] * 6.25 + t["cache_read_input_tokens"] * 0.5) / 1e6
print(json.dumps({"rows": len(rows), **t, "usd_opus": round(cost, 2)}))
