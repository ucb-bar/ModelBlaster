# args: ARM STEPS MULTITURN(0|1) SEED [FORCE_ORDER]   -- one arm, same command line as A2's ab5.sh arm(); cap $15 / 90 min
ARM=$1; STEPS=$2; MT=$3; SEED=$4; FORCE=$5
R=/tmp/krisd/mt_$ARM; cd $R; . $R/env.sh; . /tmp/krisd/farm/relay.env
KB=$R/code/cur/ModelBlaster/third_party/KernelBlaster
LED=/tmp/krisd/farm/llm_ledger.jsonl
L0=$(wc -l < $LED 2>/dev/null || echo 0); echo $L0 > $R/ledger.start
export MB_RECURRENT_CLASS=1   # (only read at export time; harmless)
if [ "$MT" = 1 ]; then export KERNELBLASTER_MULTITURN=1 KERNELBLASTER_MULTITURN_BUDGET=3,4,120; else unset KERNELBLASTER_MULTITURN; fi
if [ -n "$FORCE" ]; then export MB_FORCE_TECH_ORDER="$FORCE"; else unset MB_FORCE_TECH_ORDER; fi
rm -rf $R/out
# $15 watchdog on this node's relay ledger rows added since start
( while sleep 20; do
    s=$(tail -n +$((L0+1)) $LED | python3 -c "import sys,json;print(sum(json.loads(l).get('est_usd',0) for l in sys.stdin if l.strip()))")
    echo "$(date +%s) $s" >> $R/cost.log
    if python3 -c "import sys;sys.exit(0 if float('$s')>15 else 1)"; then echo CAP_HIT >> $R/cost.log; pkill -f "run_multislot.py $R/prob"; fi
  done ) &
WD=$!
echo "=== $ARM start $(date +%s)"
(cd $KB && timeout 5400 python scripts/run_multislot.py $R/prob --out $R/out --knowledge-dir $R/out/knowledge \
  --model azure/anthropic/claude-opus-5-5 --reflect-model azure/anthropic/claude-opus-5-5 --upkeep-model azure/anthropic/claude-opus-5-5 \
  --arch 89 --rollouts 1 --steps $STEPS --gpus 0 --knowledge v2 --ncu best --efficiency-rules off --seed $SEED > $R/run.log 2>&1)
echo "=== $ARM rc $? end $(date +%s)"
kill $WD 2>/dev/null
tail -n +$((L0+1)) $LED > $R/ledger.rows.jsonl
echo ARM_DONE
