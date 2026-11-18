. /tmp/krisd/a3/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a3; mkdir -p flows irs
L=/tmp/krisd/farm/llm_ledger.jsonl; N0=$(cat $L 2>/dev/null | wc -l); echo "ledger rows before: $N0"
IR=irs/1_MLP
[ -f $IR/.done ] || { python -m modelblaster.pipeline.extract_graph --bench-file src/1_MLP.py --bench-target-mb 0 --quant fp16 --out-dir $IR > /dev/null 2>&1 && touch $IR/.done; }
KB=/tmp/krisd/a3/code/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
C="--kb $KB --python /tmp/krisd/venv/bin/python --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 --plan-mode all --plan-top-k 3 --torch-baseline off --max-variants 0 --min-gain-pct 0.5 --rollouts 2 --steps 2"
echo "=== L1a2 base $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1a2 $C --rounds 1 > flows/L1a2.log 2>&1; echo rc $?
python ledger.py $L $N0
rm -rf flows/L1e; cp -r flows/L1a2 flows/L1e; python inject_cache.py flows/L1e; python inject_store.py flows/L1e
echo "=== L1e inject (cache + impl store) + fallback-repair requeue $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1e $C --rounds 1 --more-rounds 1 --fallback-repair requeue > flows/L1e.log 2>&1; echo rc $?
python ledger.py $L $N0
grep -h "profile_fallback\|search_fixes\|WARNING final-model" flows/L1e.log | cut -c1-400
python - <<'PY'
import json
s = json.load(open("flows/L1e/summary.json"))
for ir, x in s["irs"].items():
    print(ir, x["final_us"], x["model_e2e_us"], x.get("model_speedup"), x.get("seed_left"))
r = json.load(open("flows/L1e/r01/round.json"))
print("ingest", r.get("ingest"))
print("requeue_ingest", r.get("requeue_ingest"))
print("kb_runs", r.get("kb_runs"))
PY
echo ALLDONE $(date)
