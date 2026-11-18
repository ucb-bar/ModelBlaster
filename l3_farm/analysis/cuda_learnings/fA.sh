tail -c 1500 /tmp/krisd/cl/flows/F41.log; ls /tmp/krisd/cl/flows/F41/r00 2>/dev/null; python3 -c "
import json
d=json.load(open(\"/tmp/krisd/cl/flows/F41/graph_hints.json\"))
for h in d[\"hints\"]: print(h[\"id\"], h[\"status\"], [s.get(\"est_gain_pct\") for s in h.get(\"source\",[])])
" 2>/dev/null
