. /tmp/krisd/a3/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a3; mkdir -p flows irs
IR=irs/41_GRUBidirectional
[ -f $IR/.done ] || { python -m modelblaster.pipeline.extract_graph --bench-file data/41_GRUBidirectional/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > /dev/null 2>&1 && touch $IR/.done; }
KB=/tmp/krisd/a3/code/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
C="--kb $KB --python /tmp/krisd/venv/bin/python --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 --plan-mode all --plan-top-k 1 --torch-baseline off --max-variants 0 --min-gain-pct 0.5 --rollouts 4 --steps 4 --efficiency-rules off --rounds 1"
L=/tmp/krisd/farm/llm_ledger.jsonl; N0=$(cat $L 2>/dev/null | wc -l); echo "ledger rows before: $N0"
for A in composite lineage; do
  echo "=== L2 $A $(date)"
  python -m modelblaster.pipeline.kb_graph_opt $IR --work flows/L2_$A $C --rollout-accept $A > flows/L2_$A.log 2>&1; echo rc $?
  python ledger.py $L $N0
done
echo ALLDONE $(date)
