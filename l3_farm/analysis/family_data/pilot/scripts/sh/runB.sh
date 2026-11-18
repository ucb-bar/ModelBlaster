# B (and the smoke test): kb_graph_opt --plan-mode family on DenseNet121, conv2d_f16 family only, sweep's graph hints
# Args: <name> <rollouts> <steps> <max_llm_calls> <autotune 0|1>
. /tmp/krisd/family/sh/env.sh
NAME=$1; RO=$2; ST=$3; CAP=$4; AT=$5
WK=$W/pilot/$NAME; mkdir -p $WK; [ -f $WK/graph_hints.json ] || cp $W/in/sweep/graph_hints.json $WK/graph_hints.json
echo running > $WK/status; [ -f $WK/t0 ] || date +%s > $WK/t0; date +%s >> $WK/segments
MODEL=azure/anthropic/claude-opus-5-5
python -m modelblaster.pipeline.kb_graph_opt $W/irs/15_DenseNet121 --work $WK --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $MODEL --reflect-model $MODEL --upkeep-model $MODEL --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode family --plan-ops conv2d_f16 --min-gain-pct 0.5 --rounds 1 --rollouts $RO --steps $ST \
  --variant-steps 2 --max-variants 0 --parallel-hints auto --torch-baseline off $( [ "$AT" = 1 ] && echo "--family-autotune 1" ) \
  --kb-arg=--max-llm-calls --kb-arg=$CAP > $WK/run.log 2>&1
echo "done rc=$?" > $WK/status; date +%s > $WK/t1
