# A2 items 2 + 5: three KernelBlaster arms on 41's l1-5 reverse GRU problem (seed = shipped v1), sequential on this GPU.
#   S_gemm  class gemm (as in the sweep), single-shot codegen, 8 steps
#   S_rec   class recurrent, single-shot, 8 steps
#   M_rec   class recurrent, multi-turn with the experiment tool (budget 3 turns, 4 runs, 120 s per step), 4 steps
. /tmp/krisd/a2/env.sh
. /tmp/krisd/farm/relay.env
cd /tmp/krisd/a2 && mkdir -p ab5
KB=/tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster
for cls in gemm recurrent; do
  [ $cls = recurrent ] && export MB_RECURRENT_CLASS=1 || unset MB_RECURRENT_CLASS
  rm -rf ab5/prob_$cls; python /tmp/krisd/a2/node/mkprob.py rev b41 runs/prune41/ir_final ab5/prob_$cls
done
unset MB_RECURRENT_CLASS
arm() {  # name prob steps multiturn
  echo "=== $1 start $(date +%s)"
  if [ "$4" = 1 ]; then export KERNELBLASTER_MULTITURN=1 KERNELBLASTER_MULTITURN_BUDGET=3,4,120; else unset KERNELBLASTER_MULTITURN; fi
  rm -rf ab5/$1
  (cd $KB && timeout 5400 python scripts/run_multislot.py /tmp/krisd/a2/ab5/$2 --out /tmp/krisd/a2/ab5/$1 \
    --knowledge-dir /tmp/krisd/a2/ab5/$1/knowledge --model azure/anthropic/claude-opus-5-5 \
    --reflect-model azure/anthropic/claude-opus-5-5 --upkeep-model azure/anthropic/claude-opus-5-5 \
    --arch 89 --rollouts 1 --steps $3 --gpus 0 --knowledge v2 --ncu best --efficiency-rules off --seed 0 \
    > /tmp/krisd/a2/ab5/$1.log 2>&1)
  echo "=== $1 rc $? end $(date +%s)"
  python3 - /tmp/krisd/a2/ab5/$1 <<'PY'
import json, glob, sys
for f in glob.glob(sys.argv[1] + "/**/result.json", recursive=True):
    r = json.load(open(f))
    print(json.dumps({k: r.get(k) for k in ("baseline_partition_us", "final_partition_us", "baseline_e2e_us",
                      "final_e2e_us", "final_ok", "llm_calls", "llm_usage", "multiturn")})[:1500])
PY
}
arm S_gemm prob_gemm 8 0
arm S_rec prob_recurrent 8 0
arm M_rec prob_recurrent 4 1
echo AB5_DONE
