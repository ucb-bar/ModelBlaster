# 48_Mamba2ReturnY through the flow with the precision policy (user request): extract with --policy auto, then
# kb_graph_opt 2 rounds (rollouts 2, steps 3, plan top-k 8), global knowledge off, 2 h cap; PyTorch fp32/fp16 timing.
. /tmp/krisd/a2/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a2 && mkdir -p f48
W=f48/work; IR=f48/irs/48_Mamba2ReturnY
if [ ! -f $IR/graph.json ]; then
  python -m modelblaster.pipeline.precision_policy extract --bench-file problems/48_Mamba2ReturnY.py --out-dir $IR \
     --policy auto -- --bench-target-mb 0 > f48/extract.log 2>&1 || { echo EXTRACT_FAILED; exit 1; }
fi
grep -q @@TORCH48 f48/torch48.log 2>/dev/null || python /tmp/krisd/a2/node/torch48.py problems/48_Mamba2ReturnY.py > f48/torch48.log 2>&1
rm -rf $W
grep @@TORCH48 f48/torch48.log
KB=/tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster
echo "=== flow start $(date +%s)"
timeout 7200 python -m modelblaster.pipeline.kb_graph_opt $IR --work $W --kb $KB --python /tmp/krisd/venv/bin/python \
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 \
  --upkeep-model azure/anthropic/claude-opus-5-5 --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --plan-top-k 8 --min-gain-pct 0.5 --parallel-hints auto --torch-baseline off \
  --rounds 2 --rollouts 2 --steps 3 --variant-steps 2 --max-variants 2 --precision-policy auto > f48/flow.log 2>&1
echo "=== flow rc $? end $(date +%s)"
tail -c 3000 f48/flow.log
echo F48_DONE
