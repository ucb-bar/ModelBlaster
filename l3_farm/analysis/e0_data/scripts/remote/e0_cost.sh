# E0: spend so far of this node's relay ledger rows since <t0 epoch>, at assumed Opus prices. Args: <t0>
python3 - /tmp/krisd/farm/llm_ledger.jsonl $1 <<'PY'
import datetime, json, sys
t0 = float(sys.argv[2]); n = u = 0
for ln in open(sys.argv[1]):
    try:
        d = json.loads(ln); ts = datetime.datetime.fromisoformat(d["ts"]).timestamp()
    except Exception:
        continue
    if ts < t0:
        continue
    cr, cw, pi, o = (d.get(k) or 0 for k in ("cache_read_input_tokens", "cache_write_input_tokens", "prompt_tokens", "completion_tokens"))
    n += d.get("calls", 1) or 1
    u += (max(0, pi - cr - cw) * 5 + cr * 0.5 + cw * 6.25 + o * 25) / 1e6
print(f"calls={n} usd={u:.2f}")
PY
