# zero-LLM part, then (VIEW31_LLM=1) a 1-rollout x 3-step KernelBlaster run on the fused slot
. /tmp/krisd/a2/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a2 && rm -rf b31 && tar xzf b31.tgz 2>/dev/null
python /tmp/krisd/a2/node/view31.py /tmp/krisd/a2/b31 /tmp/krisd/a2/runs/view31 || exit 1
[ "$VIEW31_LLM" = 1 ] || exit 0
KB=/tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster
echo "=== llm start $(date +%s)"
(cd $KB && timeout 5400 python scripts/run_multislot.py /tmp/krisd/a2/runs/view31/fused --out /tmp/krisd/a2/runs/view31/run \
  --knowledge-dir /tmp/krisd/a2/runs/view31/run/knowledge --model azure/anthropic/claude-opus-5-5 \
  --reflect-model azure/anthropic/claude-opus-5-5 --upkeep-model azure/anthropic/claude-opus-5-5 \
  --arch 89 --rollouts 1 --steps 3 --gpus 0 --knowledge v2 --ncu best --efficiency-rules off --seed 0 \
  > /tmp/krisd/a2/runs/view31/run.log 2>&1)
echo "=== llm rc $? end $(date +%s)"
python3 - <<'PY'
import json, glob
for f in glob.glob("/tmp/krisd/a2/runs/view31/run/**/result.json", recursive=True):
    r = json.load(open(f))
    print(json.dumps({k: r.get(k) for k in ("baseline_partition_us", "final_partition_us", "final_ok", "llm_calls", "llm_usage")})[:1200])
PY
