. /tmp/krisd/cl/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/cl
W=flows/F11; IR=irs/11_VGG16
mkdir -p flows irs src && cp data/11_VGG16/problem.py src/11_VGG16.py
export MB_TORCH_SRC=/tmp/krisd/cl/src
echo "=== warm $(date)"
python warm.py 11_VGG16 data/11_VGG16 $IR $W --reset-hint fuse:fused__conv2d_f16__relu_f16 || exit 1
KB=/tmp/krisd/cl/code/ModelBlaster/third_party/KernelBlaster
echo "=== flow $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work $W --kb $KB --python /tmp/krisd/venv/bin/python \
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 \
  --upkeep-model azure/anthropic/claude-opus-5-5 --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --min-gain-pct 0.5 --parallel-hints auto --torch-baseline off --rounds 1 --rollouts 2 --steps 2 --variant-steps 3 --max-variants 3 --fusion-extended epilogue > $W.log 2>&1
echo "=== rc $? $(date)"
tail -c 4000 $W.log

