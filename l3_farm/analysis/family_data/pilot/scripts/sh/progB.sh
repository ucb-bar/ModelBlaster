WK=/tmp/krisd/family/pilot/${1:-B}
cat $WK/status
J=$(ls $WK/r00/run/kb_*/journal/state.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - $J <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
st=s.get("steps_log",[])
best=s.get("best",{}).get("conv2d_f16",{})
print("steps",len(st),"llm_calls",s.get("llm_calls"),"best slot_us",best.get("slot_us"),"e2e",s.get("e2e_us"))
for x in st[-6:]:
    g=(x["context"].get("e2e") or {})
    print(x["rollout"],x["step"],x["technique"][:28],x["status"],round(x["slot_us_after"] or 0,1),"fix",x["fix_attempts"],"guard" if g.get("family_guard") else "", "acc" if g.get("accept") else "")
PY
tail -3 /tmp/krisd/family/logs/matrix.log
