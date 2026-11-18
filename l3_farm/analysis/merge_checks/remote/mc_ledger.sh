# merge checks: copy this node's relay ledger rows of one run window into the run dir. Args: <run dir>
R=$1; T0=$(cat $R/t0); T1=$(cat $R/t1 2>/dev/null || date +%s)
python3 - /tmp/krisd/farm/llm_ledger.jsonl $T0 $T1 > $R/ledger.jsonl <<'PY'
import datetime, json, sys
t0 = float(sys.argv[2]); t1 = float(sys.argv[3])
try:
    lines = open(sys.argv[1]).read().splitlines()
except OSError:
    lines = []
for ln in lines:
    try:
        d = json.loads(ln); ts = datetime.datetime.fromisoformat(d["ts"]).timestamp()
    except Exception:
        continue
    if t0 <= ts <= t1:
        print(ln)
PY
wc -l < $R/ledger.jsonl
