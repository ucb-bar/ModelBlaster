# Long-lived (detach). Args: <stamp> <arm> <problem> [kb_graph_opt flags...]. One A/B arm on the frozen tree:
# extract the IR (native size, fp16), then kb_graph_opt with the sweep's flags at reduced rollouts + the arm's flags.
# Status in /tmp/krisd/a4/runs/<arm>/status; ledger offset recorded at start (/tmp/krisd/farm/llm_ledger.jsonl).
STAMP=$1; ARM=$2; NAME=$3; shift 3
A=/tmp/krisd/a4/$STAMP; D=/tmp/krisd/a4/runs/$ARM; mkdir -p $D
st(){ echo "$1" > $D/status; }
st running
wc -l < /tmp/krisd/farm/llm_ledger.jsonl > $D/ledger_start 2>/dev/null || echo 0 > $D/ledger_start
date +%s > $D/t_start
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$A MB_TORCH_SRC=$D/src
. /tmp/krisd/farm/relay.env
KB=$A/ModelBlaster/third_party/KernelBlaster
cd $A
MODEL=azure/anthropic/claude-opus-5-5
mkdir -p $D/src; cp /tmp/krisd/a4/problems/$NAME.py $D/src/$NAME.py
IR=$D/irs/$NAME
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $D/src/$NAME.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $D/extract.log 2>&1 \
    || { st failed:extract; exit 1; }
  touch $IR/.done
fi
timeout 5h python -m modelblaster.pipeline.kb_graph_opt $IR --work $D/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $MODEL --reflect-model $MODEL --upkeep-model $MODEL --knowledge v2 --ncu best --gpus 0 --arch 89 \
  --plan-mode all --plan-top-k 32 --plan-min-share 0.002 --min-gain-pct 0.5 --rounds 3 --rollouts 2 --steps 4 \
  --variant-steps 2 --max-variants 4 --parallel-hints auto "$@" > $D/run.log 2>&1
rc=$?
date +%s > $D/t_end
wc -l < /tmp/krisd/farm/llm_ledger.jsonl > $D/ledger_end 2>/dev/null
if [ $rc = 124 ]; then st failed:timeout; elif [ $rc = 0 ]; then st done; else st failed:exit$rc; fi
