# merge checks: one run end to end from a frozen tree. Args: <run dir> (absolute, under /tmp/krisd/mc/runs/...).
# <run dir>/run.env (written by the driver) sets: LABEL PROBLEM SEED STUB(0|1) NCU BUDGET FLAGS ENVS TH ARMNAME
# <run dir>/problem.py is uploaded. Global knowledge is always OFF. Status in <run dir>/status.
R=$1; . $R/run.env
E=/tmp/krisd/mc; CR=$E/$LABEL; KB=$CR/ModelBlaster/third_party/KernelBlaster
st(){ echo "$1" > $R/status; }
bash -c "exec -a mc_hold_kb_graph_opt_$PROBLEM sleep 86400" & HOLD=$!      # keeps the farm dispatcher's probe at foreign!=0
st running; date +%s > $R/t0
export PATH=$CUDA_N/bin:$PATH CUDA_HOME=$CUDA_N MB_CUPTI_LIB=$CUDA_N/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR MB_TORCH_SRC=$R/src
[ -f /tmp/krisd/farm/relay.env ] && . /tmp/krisd/farm/relay.env
[ -n "$ENVS" ] && export $ENVS
SMIQ=name,uuid,driver_version,vbios_version,temperature.gpu,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,utilization.gpu
{ echo "job=${SLURM_JOB_ID:-?} host=$(hostname)"; nvidia-smi --query-gpu=$SMIQ --format=csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > $R/smi_start.txt 2>&1
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader)"; sleep 120; done ) > $R/smi_samples.txt 2>&1 & SMI=$!
cd $CR
IR=$R/irs/$PROBLEM; mkdir -p $R/src; cp $R/problem.py $R/src/$PROBLEM.py
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $R/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $R/extract.log 2>&1 \
    || { st failed:extract; kill $HOLD $SMI; exit 1; }
  touch $IR/.done
fi
date +%s > $R/t_extract
M=$MODEL; PL=""
if [ "$STUB" = 1 ]; then       # zero-LLM: every slot comes back unchanged (stub:graphopt, identity), so keys / manifests / plans are LLM-free
  M=stub:graphopt; PL="--plugin $R/plugin.py"
  cat > $R/plugin.py <<'PY'
from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm
def scripted(msgs):
    user = msgs[-1]["content"]
    cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user else user.split("Current slot source:")[-1])
    return "```cuda\n" + (cur or "") + "```"
agentic_llm.register_stub("graphopt", scripted)
PY
fi
echo "python -m modelblaster.pipeline.kb_graph_opt $IR --work $R/work --kb $KB --model $M $PL --knowledge v2 --ncu $NCU --gpus 0 --arch 89 --plan-mode all --plan-top-k 32 --plan-min-share 0.002 --min-gain-pct 0.5 $BUDGET --variant-steps 2 --max-variants 4 --parallel-hints auto --knowledge-global off --seed $SEED $FLAGS | ENVS=$ENVS" > $R/cmdline.txt
timeout ${TH}h python -m modelblaster.pipeline.kb_graph_opt $IR --work $R/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $M --reflect-model $M --upkeep-model $M $PL --knowledge v2 --ncu $NCU --gpus 0 --arch 89 \
  --plan-mode all --plan-top-k 32 --plan-min-share 0.002 --min-gain-pct 0.5 $BUDGET \
  --variant-steps 2 --max-variants 4 --parallel-hints auto --knowledge-global off --seed $SEED $FLAGS > $R/run.log 2>&1
rc=$?; date +%s > $R/t1
if [ $rc = 124 ]; then FS=failed:timeout; elif [ $rc = 0 ]; then FS=done; else FS=failed:exit$rc; fi
echo $FS > $R/flow_status
st remeasuring
# quiet re-measure on this node, nothing else on the GPU: the run's final cache + hints, CUDA-graph replay, torch baseline
# the deployed program's export-time settings (concurrency is an env var at export, not part of the cache): replicate the flow's last choice
CONC=$(python3 -c "import json,sys
try:
    c = json.load(open('$R/work/summary.json')).get('cuda_learnings') or {}
    v = c.get('conc_auto_choice') or c.get('graph_concurrency') or 'off'
    print('on' if v == 'auto' else v)
except Exception:
    print('off')")
[ "$CONC" != off ] && export MB_GRAPH_CONCURRENCY=$CONC && echo "remeasure with MB_GRAPH_CONCURRENCY=$CONC" > $R/remeasure.env
if [ -d $R/work/cache ]; then
  python -m modelblaster.pipeline.kb_model_run $IR --cache $R/work/cache --hints $R/work/graph_hints.json --arch 89 \
    --gpu-model "NVIDIA L40S" --reps 5 --work $R/remeasure --out $R/remeasure.json --evaluator local --kb $KB \
    --kb-python /tmp/krisd/venv/bin/python --gpus 0 --torch-baseline $([ "$STUB" = 1 ] && echo off || echo auto) > $R/remeasure.log 2>&1
fi
{ echo "job=${SLURM_JOB_ID:-?} host=$(hostname)"; nvidia-smi --query-gpu=$SMIQ --format=csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } > $R/smi_end.txt 2>&1
date +%s > $R/t2
st $FS
kill $HOLD $SMI 2>/dev/null
