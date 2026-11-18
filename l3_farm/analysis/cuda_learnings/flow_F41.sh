. /tmp/krisd/cl/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/cl
W=flows/F41; IR=irs/41_GRUBidirectional
mkdir -p flows irs src && cp data/41_GRUBidirectional/problem.py src/41_GRUBidirectional.py
export MB_TORCH_SRC=/tmp/krisd/cl/src
echo "=== warm $(date)"
python warm.py 41_GRUBidirectional data/41_GRUBidirectional $IR $W  || exit 1
KB=/tmp/krisd/cl/code/ModelBlaster/third_party/KernelBlaster
echo "=== flow $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work $W --kb $KB --python /tmp/krisd/venv/bin/python \
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 \
  --upkeep-model azure/anthropic/claude-opus-5-5 --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --min-gain-pct 0.5 --parallel-hints auto --torch-baseline off --rounds 1 --rollouts 2 --steps 2 --variant-steps 3 --max-variants 2 --reuse-winners siblings --graph-concurrency on --fusion-extended siblings > $W.log 2>&1
echo "=== rc $? $(date)"
tail -c 4000 $W.log

