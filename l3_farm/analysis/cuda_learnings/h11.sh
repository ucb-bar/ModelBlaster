python3 -c "
import json
d=json.load(open(\"/tmp/krisd/cl/flows/F11/graph_hints.json\"))
for h in d[\"hints\"]:
    print(h[\"id\"], h[\"status\"])
    for e in h.get(\"evidence\",[]): print(\"   \", e.get(\"decision\"), e.get(\"gain\"), e.get(\"original_us\"), e.get(\"variant_us\"), (e.get(\"remeasured\") or {}).get(\"original_us\"), (e.get(\"remeasured\") or {}).get(\"variant_us\"), e.get(\"why\"))
"
