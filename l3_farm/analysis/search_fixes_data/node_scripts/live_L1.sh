. /tmp/krisd/a3/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a3; mkdir -p flows irs
IR=irs/1_MLP
[ -f $IR/.done ] || { python -m modelblaster.pipeline.extract_graph --bench-file src/1_MLP.py --bench-target-mb 0 --quant fp16 --out-dir $IR > /dev/null 2>&1 && touch $IR/.done; }
KB=/tmp/krisd/a3/code/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
C="--kb $KB --python /tmp/krisd/venv/bin/python --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 --plan-mode all --plan-top-k 3 --torch-baseline off --max-variants 0 --min-gain-pct 0.5 --rollouts 2 --steps 2"
L=/tmp/krisd/farm/llm_ledger.jsonl; N0=$(cat $L 2>/dev/null | wc -l); echo "ledger rows before: $N0"
echo "=== L1a base $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1a $C --rounds 1 > flows/L1a.log 2>&1; echo rc $?
python ledger.py $L $N0
for V in b c; do rm -rf flows/L1$V; cp -r flows/L1a flows/L1$V; python inject_cache.py flows/L1$V; done
echo "=== L1b inject + fallback-repair on (default) $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1b $C --rounds 1 --more-rounds 1 > flows/L1b.log 2>&1; echo rc $?
python ledger.py $L $N0
echo "=== L1c inject + fallback-repair requeue $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L1c $C --rounds 1 --more-rounds 1 --fallback-repair requeue > flows/L1c.log 2>&1; echo rc $?
python ledger.py $L $N0
grep -h "profile_fallback\|search_fixes\|WARNING final-model" flows/L1b.log flows/L1c.log | cut -c1-600
echo ALLDONE $(date)
