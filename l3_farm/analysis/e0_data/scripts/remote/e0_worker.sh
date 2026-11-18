# E0 noise floor: one replicate end to end from the frozen tree in /tmp/krisd/e0 (worker4.sh recipe, GK off, --seed).
# Args: <problem> <seed> [timeout_h]. Run dir /tmp/krisd/e0/runs/<problem>_s<seed> (problem.py + bench.py uploaded).
P=$1; SEED=$2; TH=${3:-4}; E=/tmp/krisd/e0; CR=$E; R=$E/runs/${P}_s$SEED
st(){ echo "$1" > $R/status; }
# keep the farm dispatcher's probe counting this node as foreign (busy) for the whole replicate incl. re-measure
bash -c "exec -a e0_hold_kb_graph_opt_$P sleep 86400" & HOLD=$!
st running; date +%s > $R/t0
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR MB_TORCH_SRC=$R/src
. /tmp/krisd/farm/relay.env
KB=$CR/ModelBlaster/third_party/KernelBlaster
M=azure/anthropic/claude-opus-5-5
cd $CR
IR=$R/irs/$P; mkdir -p $R/src; cp $R/problem.py $R/src/$P.py
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $R/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $R/extract.log 2>&1 \
    || { st failed:extract; kill $HOLD; exit 1; }
  touch $IR/.done
fi
date +%s > $R/t_extract
timeout ${TH}h python -m modelblaster.pipeline.kb_graph_opt $IR --work $R/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $M --reflect-model $M --upkeep-model $M --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --plan-top-k 32 --plan-min-share 0.002 --min-gain-pct 0.5 --rounds 3 --rollouts 4 --steps 4 \
  --variant-steps 2 --max-variants 4 --parallel-hints auto --knowledge-global off --seed $SEED > $R/run.log 2>&1
rc=$?; date +%s > $R/t1
if [ $rc = 124 ]; then FS=failed:timeout; elif [ $rc = 0 ]; then FS=done; else FS=failed:exit$rc; fi
echo $FS > $R/flow_status
st remeasuring
# quiet re-measure: nothing else runs on this (1-GPU) node now. Our model from the run's final cache + hints,
# CUDA-graph replay, with the flow's PyTorch comparison; then the standalone torch_posthoc bench (eager/graph/compile).
if [ -d $R/work/cache ]; then
  python -m modelblaster.pipeline.kb_model_run $IR --cache $R/work/cache --hints $R/work/graph_hints.json --arch 89 \
    --gpu-model "NVIDIA L40S" --reps 5 --work $R/remeasure --out $R/remeasure.json --evaluator local --kb $KB \
    --kb-python /tmp/krisd/venv/bin/python --gpus 0 --torch-baseline auto > $R/remeasure.log 2>&1
fi
python $R/bench.py $R/problem.py $P 50 10 1 > $R/posthoc.log 2>&1
grep '^@@POSTHOC@@' $R/posthoc.log | sed 's/^@@POSTHOC@@//' > $R/posthoc.json
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $R/gpu_apps_end.txt 2>&1
date +%s > $R/t2
st $FS
kill $HOLD
