# Args: <stamp> <arm dirs...>. Each arm's final model (its cache + accepted hints) timed on THIS node, arms interleaved x2,
# kb_model_run --reps 5 (CUDA-graph replay); then one torch baseline. Output lines: RESULT <arm> <rep> <model_e2e_us>.
STAMP=$1; shift
A=/tmp/krisd/a4/$STAMP
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$A
cd $A
for rep in 1 2; do for D in "$@"; do
  N=$(basename $D)
  python -m modelblaster.pipeline.kb_model_run $D/irs/8_ResNetBasicBlock --cache $D/work/cache --hints $D/work/graph_hints.json \
    --arch 89 --reps 5 --torch-baseline off --work /tmp/krisd/a4/remeasure/$N.$rep --out /tmp/krisd/a4/remeasure/$N.$rep.json > /tmp/krisd/a4/remeasure/$N.$rep.log 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT', sys.argv[2], sys.argv[3], d.get('model_e2e_us'), d.get('passed'))" /tmp/krisd/a4/remeasure/$N.$rep.json $N $rep
done; done
