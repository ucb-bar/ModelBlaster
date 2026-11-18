python3 -c "
import json
r=json.load(open(\"/tmp/krisd/cl/runs/41_GRUBidirectional.siblings/gpu_tests.json\"))
for k,v in r.items():
    if isinstance(v, dict): print(k, {kk: vv for kk, vv in v.items() if kk not in (\"log_tail\",)})
    else: print(k, v)
print(r[\"watchdog\"][\"log_tail\"])
"
