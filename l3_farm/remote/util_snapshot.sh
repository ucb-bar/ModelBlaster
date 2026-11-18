# Read-only snapshot: CPU load, GPU utilisation (6 samples over ~5 s), and the relay ledger's recent activity (if any). Prints no secrets.
echo "host=$(hostname) nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg) memfree_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
echo "gpu_util=$(for i in 1 2 3 4 5 6; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';'; sleep 1; done)"
L=/tmp/krisd/farm/llm_ledger.jsonl
if [ -f $L ]; then
  . /tmp/krisd/venv/bin/activate 2>/dev/null
  python3 - $L <<'P'
import json,sys,time
rows=[]
for ln in open(sys.argv[1]):
    try: rows.append(json.loads(ln))
    except Exception: pass
print("ledger rows", len(rows), "keys", sorted(rows[-1].keys())[:20] if rows else None)
ts=[r.get("ts") or r.get("time") or r.get("t") for r in rows]
now=time.time()
for w in (300,900):
    rec=[r for r,t in zip(rows,ts) if isinstance(t,(int,float)) and now-t<=w]
    print(f"last {w}s: {len(rec)} calls", "cost", round(sum((r.get('cost_usd') or r.get('usd') or 0) for r in rec),2), "tok_in", sum((r.get('prompt_tokens') or r.get('tokens_in') or 0) for r in rec), "tok_out", sum((r.get('completion_tokens') or r.get('tokens_out') or 0) for r in rec))
P
fi
