# A2 item 4 with LLM: KernelBlaster on 41's whole model as one rnn_stack region slot (forward winner for every GRU,
# level-parallel composite seed: 3,890 us; the per-op graph with concurrent branches: 3,635 us). 1 rollout x 6 steps.
. /tmp/krisd/a2/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a2 && rm -rf b41 && tar xzf b41.tgz 2>/dev/null
[ -f runs/prune41/ir_final/graph.json ] || python /tmp/krisd/a2/node/prune41.py /tmp/krisd/a2/b41 /tmp/krisd/a2/runs/prune41 1 > logs/prune41_220.log 2>&1
mkdir -p reg && rm -rf reg/prob reg/prob_ir reg/run
MB_RECURRENT_CLASS=1 python /tmp/krisd/a2/node/mkprob.py region b41 runs/prune41/ir_final reg/prob
KB=/tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster
echo "=== start $(date +%s)"
(cd $KB && timeout 7200 python scripts/run_multislot.py /tmp/krisd/a2/reg/prob --out /tmp/krisd/a2/reg/run \
  --knowledge-dir /tmp/krisd/a2/reg/run/knowledge --model azure/anthropic/claude-opus-5-5 \
  --reflect-model azure/anthropic/claude-opus-5-5 --upkeep-model azure/anthropic/claude-opus-5-5 \
  --arch 89 --rollouts 1 --steps 6 --gpus 0 --knowledge v2 --ncu best --efficiency-rules off --seed 0 \
  > /tmp/krisd/a2/reg/run.log 2>&1)
echo "=== rc $? end $(date +%s)"
python3 - <<'PY'
import json, glob
for f in glob.glob("/tmp/krisd/a2/reg/run/**/result.json", recursive=True):
    r = json.load(open(f))
    print(json.dumps({k: r.get(k) for k in ("baseline_partition_us", "final_partition_us", "baseline_e2e_us",
                      "final_e2e_us", "final_ok", "llm_calls", "llm_usage")})[:1500])
PY
echo REG_DONE
