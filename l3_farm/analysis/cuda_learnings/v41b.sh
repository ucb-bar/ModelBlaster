cd /tmp/krisd/cl/flows/F41/r00/variants/fuse_*; grep -h "baseline\|store seed\|seed candidate" run.log | head; python3 -c "
import json
m=json.load(open(\"orig/manifest.json\")); print(m.get(\"concurrency\")); print([o[\"name\"] for o in m[\"ops\"]])
m=json.load(open(\"variant/manifest.json\")); print(m.get(\"concurrency\")); print([o[\"name\"] for o in m[\"ops\"]])"
