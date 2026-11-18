. /tmp/krisd/a3/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a3
L=/tmp/krisd/farm/llm_ledger.jsonl; N0=$(cat $L 2>/dev/null | wc -l); echo "ledger rows before: $N0"
IR=irs/1_MLP
KB=/tmp/krisd/a3/code/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
C="--kb $KB --python /tmp/krisd/venv/bin/python --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 --plan-mode all --plan-top-k 3 --torch-baseline off --max-variants 0 --min-gain-pct 0.5 --rollouts 2 --steps 2"
rm -rf flows/L1e; cp -r flows/L1a2 flows/L1e; python inject_cache.py flows/L1e && python inject_store.py flows/L1e || exit 1
echo "=== L1e inject (cache + impl store) + fallback-repair requeue $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1e $C --rounds 1 --more-rounds 1 --fallback-repair requeue > flows/L1e.log 2>&1; echo rc $?
python ledger.py $L $N0
grep -h "profile_fallback\|search_fixes\|WARNING final-model" flows/L1e.log | cut -c1-300
python - <<'PY'
import json
s = json.load(open("flows/L1e/summary.json"))
for ir, x in s["irs"].items():
    print(ir, x["final_us"], x["model_e2e_us"], x.get("model_speedup"), x.get("seed_left"))
r = json.load(open("flows/L1e/r01/round.json"))
print("ingest", r.get("ingest"))
print("requeue_ingest", r.get("requeue_ingest"))
print("kb_runs", r.get("kb_runs"))
w = json.load(open("flows/L1e/r01/winners/winners.json"))
print("winners export (shape 16384x16384):", [(x["id"], x["slot_us"]) for x in w["winners"] if x["op"] == "linear_f16" and x["shapes"][0].get("N") == 16384])
PY
echo ALLDONE $(date)
