# One problem end to end: extract IR at native size (no scaling), then kb_graph_opt. Status in $D/status.
NAME=$1; D=/tmp/krisd/farm/$NAME; . $D/job.env
st(){ echo "$1" > $D/status; }
st running
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=/tmp/krisd MB_TORCH_SRC=$D/src
. /tmp/krisd/farm/relay.env
KB=/tmp/krisd/ModelBlaster/third_party/KernelBlaster
cd /tmp/krisd
IR=$D/irs/$NAME; mkdir -p $D/src; cp $D/problem.py $D/src/$NAME.py   # torch baseline finds <ir name>.py under $MB_TORCH_SRC
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $D/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $D/extract.log 2>&1 \
    || { st failed:extract; exit 1; }
  touch $IR/.done
fi
timeout ${TIMEOUT_H}h python -m modelblaster.pipeline.kb_graph_opt $IR --work $D/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $MODEL --reflect-model $MODEL --upkeep-model $MODEL --knowledge v2 --ncu best --gpus 0 --arch $ARCH \
  --plan-mode all --plan-top-k $PLAN_TOPK --plan-min-share $PLAN_MINSHARE --min-gain-pct $MIN_GAIN --rounds $ROUNDS --rollouts $ROLLOUTS --steps $STEPS \
  --variant-steps $VSTEPS --max-variants $MAXVAR --parallel-hints auto > $D/run.log 2>&1
rc=$?
if [ $rc = 124 ]; then st failed:timeout; elif [ $rc = 0 ]; then st done; else st failed:exit$rc; fi
