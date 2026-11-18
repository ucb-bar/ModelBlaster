#!/usr/bin/env python3
"""Write a node-side flow script: warm-start NAME from the sweep and run kb_graph_opt with the given flags.
  mkflow.py TAG NAME "WARM_ARGS" "FLOW_FLAGS" > flow_TAG.sh"""
import sys
tag, name, warm, flags = sys.argv[1:5]
print(f""". /tmp/krisd/cl/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/cl
W=flows/{tag}; IR=irs/{name}
mkdir -p flows irs src && cp data/{name}/problem.py src/{name}.py
export MB_TORCH_SRC=/tmp/krisd/cl/src
echo "=== warm $(date)"
python warm.py {name} data/{name} $IR $W {warm} || exit 1
KB=/tmp/krisd/cl/code/ModelBlaster/third_party/KernelBlaster
echo "=== flow $(date)"
python -m modelblaster.pipeline.kb_graph_opt $IR --work $W --kb $KB --python /tmp/krisd/venv/bin/python \\
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 \\
  --upkeep-model azure/anthropic/claude-opus-5-5 --knowledge v2 --ncu best --gpus 0 --arch 89 \\
  --plan-mode all --min-gain-pct 0.5 --parallel-hints auto --torch-baseline off {flags} > $W.log 2>&1
echo "=== rc $? $(date)"
tail -c 4000 $W.log
""")
