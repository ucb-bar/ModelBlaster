# Node side: progress of one arm. Args: <tag>
D=/tmp/krisd/sonnet_$1/43_MinGPTCausalAttention
echo "status=$(cat $D/status 2>/dev/null) start=$(cat $D/t_start 2>/dev/null) end=$(cat $D/t_end 2>/dev/null) now=$(date +%s)"
echo "ledger_start=$(cat $D/ledger_start 2>/dev/null) ledger_now=$(wc -l < /tmp/krisd/farm/llm_ledger.jsonl)"
pgrep -af "kb_graph_opt|run_multislot" | grep sonnet_$1 | cut -c1-200 | head -3
tail -4 $D/run.log 2>/dev/null | cut -c1-300
tail -3 $D/worker.out 2>/dev/null | cut -c1-300
ls $D/work 2>/dev/null | tr '\n' ' '; echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
/tmp/krisd/venv/bin/python - $D <<'PY'
import json, sys, pathlib
L = pathlib.Path("/tmp/krisd/farm/llm_ledger.jsonl")
D = pathlib.Path(sys.argv[1])
try:
    n0 = int((D / "ledger_start").read_text().split()[0])
except Exception:
    n0 = 0
rows = [json.loads(l) for l in (L.read_text().splitlines() if L.exists() else [])[n0:] if l.strip()]
P = {"opus": (5, 25, 6.25, 0.5), "haiku": (0.10, 0.50, 0.125, 0.01), "sonnet": (2, 10, 2.5, 0.2)}
agg = {}
for r in rows:
    m = r.get("model", "?")
    a = agg.setdefault(m, [0, 0, 0, 0, 0])
    a[0] += 1
    for i, k in enumerate(("prompt_tokens", "completion_tokens", "cache_write_input_tokens", "cache_read_input_tokens")):
        a[i + 1] += int(r.get(k) or 0)
tot = 0
for m, a in agg.items():
    p = next((v for k, v in P.items() if k in m), P["opus"])
    usd = (a[1] * p[0] + a[2] * p[1] + a[3] * p[2] + a[4] * p[3]) / 1e6
    tot += usd
    print("ledger", m, "calls", a[0], "in", a[1], "out", a[2], "cw", a[3], "cr", a[4], f"usd {usd:.2f}")
print(f"ledger total usd {tot:.2f}; keys of a row: {sorted(rows[-1].keys()) if rows else None}")
PY
