python3 -c "
import json
r=json.load(open(\"/tmp/krisd/cl/flows/S41/r00/round.json\"))
print(json.dumps(r.get(\"kb_runs\"))[:1500]); print(json.dumps(r.get(\"reuse\"),default=str)[:1500])"
grep -h "seed candidate" /tmp/krisd/cl/flows/S41/r00/run.log | cut -c60-300 | head
