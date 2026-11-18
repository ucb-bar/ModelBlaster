. /tmp/krisd/cl/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/cl
W=flows/S41; IR=irs/41_GRUBidirectional
rm -rf $W
[ -f $IR/.done ] || { python -m modelblaster.pipeline.extract_graph --bench-file data/41_GRUBidirectional/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > /dev/null 2>&1 && touch $IR/.done; }
KB=/tmp/krisd/cl/code/ModelBlaster/third_party/KernelBlaster
echo "=== flow $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work $W --kb $KB --python /tmp/krisd/venv/bin/python \
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 \
  --upkeep-model azure/anthropic/claude-opus-5-5 --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --min-gain-pct 0.5 --parallel-hints auto --torch-baseline off --rounds 1 --rollouts 2 --steps 3 \
  --variant-steps 2 --max-variants 0 --reuse-winners siblings --graph-concurrency on > $W.log 2>&1
echo "=== rc $? $(date)"
echo ALLDONE
