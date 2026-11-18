ls /tmp/krisd/cl/flows/F11/r00 2>/dev/null; cat /tmp/krisd/cl/flows/F11/r00/progress.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted(d[\"phases\"]))"; python3 -c "
import json
d=json.load(open(\"/tmp/krisd/cl/flows/F11/graph_hints.json\"))
for h in d[\"hints\"]: print(h[\"id\"], h[\"status\"], [s.get(\"est_gain_pct\") for s in h.get(\"source\",[])], len(h.get(\"chains\") or []))
"; ls /tmp/krisd/cl/flows/F11/r00/variants 2>/dev/null; wc -l /tmp/krisd/farm/llm_ledger.jsonl 2>/dev/null
