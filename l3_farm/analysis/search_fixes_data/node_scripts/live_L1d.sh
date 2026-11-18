. /tmp/krisd/a3/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a3
L=/tmp/krisd/farm/llm_ledger.jsonl; N0=$(cat $L 2>/dev/null | wc -l); echo "ledger rows before: $N0"
IR=irs/1_MLP
KB=/tmp/krisd/a3/code/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
C="--kb $KB --python /tmp/krisd/venv/bin/python --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 --plan-mode all --plan-top-k 3 --torch-baseline off --max-variants 0 --min-gain-pct 0.5 --rollouts 2 --steps 2"
rm -rf flows/L1d; cp -r flows/L1a flows/L1d; python inject_cache.py flows/L1d
echo "=== L1d inject + fallback-repair requeue (fixed cache add) $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1d $C --rounds 1 --more-rounds 1 --fallback-repair requeue > flows/L1d.log 2>&1; echo rc $?
python ledger.py $L $N0
grep -h "profile_fallback\|search_fixes\|WARNING final-model" flows/L1d.log | cut -c1-400
python - <<'PY'
import json
s = json.load(open("flows/L1d/summary.json"))
for ir, x in s["irs"].items():
    print(ir, x["final_us"], x["model_e2e_us"], x.get("model_speedup"), x.get("seed_left"))
r = json.load(open("flows/L1d/r01/round.json"))
print(r.get("ingest"), r.get("kb_runs"))
PY
echo "=== repair 15 $(date)"
export MB_PRUNE_DEAD=0
python a3_node.py repair 15_DenseNet121 data/15_DenseNet121 runs/15.repair 2>&1 | tail -40
echo ALLDONE $(date)
