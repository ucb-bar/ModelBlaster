# Args: <from line> [to line]. Token totals of the relay ledger lines (from, to], by model; first line's keys.
python3 - "$@" <<'PY'
import json, sys, collections
f = "/tmp/krisd/farm/llm_ledger.jsonl"
a = int(sys.argv[1]); b = int(sys.argv[2]) if len(sys.argv) > 2 else None
lines = open(f).read().splitlines()[a:b]
tot = collections.defaultdict(lambda: collections.Counter())
keys = None
for l in lines:
    try: d = json.loads(l)
    except ValueError: continue
    keys = keys or sorted(d.keys())
    m = d.get("model", "?")
    tot[m]["calls"] += 1
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            tot[m][k] += v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    tot[m][k + "." + kk] += vv
print(json.dumps({"n": len(lines), "keys": keys, "by_model": {m: dict(c) for m, c in tot.items()}}, indent=1))
PY
