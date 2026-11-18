# E0: copy this node's relay ledger lines with ts >= <t0> into the run dir. Args: <run name> <t0 epoch>
L=/tmp/krisd/farm/llm_ledger.jsonl; R=/tmp/krisd/e0/runs/$1
python3 - $L $2 > $R/ledger.jsonl <<'PY'
import datetime, json, sys
t0 = float(sys.argv[2])
for ln in open(sys.argv[1]):
    try:
        d = json.loads(ln)
    except ValueError:
        continue
    ts = d.get("ts") or 0
    if isinstance(ts, str):
        try:
            ts = datetime.datetime.fromisoformat(ts).timestamp()   # naive local time of this node
        except ValueError:
            ts = 0
    if ts >= t0:
        print(ln.rstrip())
PY
wc -l < $R/ledger.jsonl
