# E0: per-round LLM cost from state.json telemetry (assumed Opus prices) + ledger roles. Args: <run name> <t0>
R=/tmp/krisd/e0/runs/$1
python3 - $R/work/state.json /tmp/krisd/farm/llm_ledger.jsonl $2 <<'PY'
import json, sys, datetime, collections
P = lambda b: (b.get("uncached_input_tokens",0)*5 + b.get("cache_read_input_tokens",0)*.5 + b.get("cache_write_input_tokens",0)*6.25 + b.get("output_tokens",0)*25)/1e6
try:
    st = json.load(open(sys.argv[1]))
    for r in st["rounds"]:
        llm = ((r.get("telemetry") or {}).get("kb") or {}).get("llm") or {}
        print("round", r["round"], "secs", r.get("seconds"), "calls", (llm.get("total") or {}).get("calls"), "usd %.2f" % sum(P(b) for b in llm.get("by_role_model", [])), "part", r.get("partition_us"), "dec", [(d.get("hint"), d.get("status")) for d in r.get("decisions", [])])
except Exception as e:
    print("state", e)
t0 = float(sys.argv[3]); roles = collections.Counter(); usd = collections.Counter()
for ln in open(sys.argv[2]):
    try:
        d = json.loads(ln); ts = datetime.datetime.fromisoformat(d["ts"]).timestamp()
    except Exception:
        continue
    if ts < t0: continue
    role = next((w.split("=",1)[1] for w in d.get("what","").split() if w.startswith("role=")), "?")
    cr, cw, pi, o = (d.get(k) or 0 for k in ("cache_read_input_tokens","cache_write_input_tokens","prompt_tokens","completion_tokens"))
    roles[role] += 1; usd[role] += (max(0, pi-cr-cw)*5 + cr*.5 + cw*6.25 + o*25)/1e6
print("ledger", {k: (roles[k], round(usd[k], 2)) for k in roles})
PY
tail -3 $R/run.log | cut -c1-400
