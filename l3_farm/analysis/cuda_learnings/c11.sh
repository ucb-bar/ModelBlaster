python3 -c "
import json
m=json.load(open(\"/tmp/krisd/cl/flows/F11/cache/index.json\"))
from collections import Counter
print(Counter(e[\"op\"] for e in m[\"entries\"]))"
