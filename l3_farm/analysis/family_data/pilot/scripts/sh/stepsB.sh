J=$(ls /tmp/krisd/family/pilot/B/r00/run/kb_*/journal/state.json | head -1)
python3 - $J <<'PY'
import json,sys,collections
s=json.load(open(sys.argv[1]))
st=s["steps_log"]
print(collections.Counter(x["status"] for x in st))
print(collections.Counter(x["technique"] for x in st).most_common(12))
for x in st:
    g=x["context"].get("e2e") or {}
    print(x["rollout"],x["step"],x["technique"][:30],x["status"],round(x["slot_us_after"] or 0,1),g.get("gain"),"GUARD" if g.get("family_guard") else "", "ACC" if g.get("accept") else "")
PY
