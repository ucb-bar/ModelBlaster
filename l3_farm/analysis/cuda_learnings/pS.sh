ls /tmp/krisd/cl/flows/S41/r00/ 2>/dev/null; python3 -c "
import json
p=json.load(open(\"/tmp/krisd/cl/flows/S41/r00/plan.json\"))
for pr in p[\"problems\"]: print(pr[\"id\"], pr[\"op_names\"])
" 2>&1 | tail -5; grep -h "baseline\|seed cand" /tmp/krisd/cl/flows/S41/r00/run.log 2>/dev/null | cut -c60-300
