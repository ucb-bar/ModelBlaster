# merge checks: re-time one collected final model (cache + hints uploaded to <rundir>) on THIS node with the tree <label>.
# Args: <label> <rundir> <problem> <conc: off|on|...> <tag>.  -> <rundir>/rem_<tag>.json (kb_model_run output). No LLM, no relay.
L=$1; D=$2; P=$3; CONC=$4; TAG=$5; E=/tmp/krisd/mc; CR=$E/$L; KB=$CR/ModelBlaster/third_party/KernelBlaster
export PATH=$CUDA_N/bin:$PATH CUDA_HOME=$CUDA_N MB_CUPTI_LIB=$CUDA_N/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR MB_TORCH_SRC=$D/src
[ -n "$ENVS" ] && export $ENVS
[ "$CONC" != off ] && export MB_GRAPH_CONCURRENCY=$CONC
IR=$E/rem_ir/$L/$P
mkdir -p $D/src && cp $D/problem.py $D/src/$P.py
if [ ! -f $IR/.done ]; then
  cd $CR && python -m modelblaster.pipeline.extract_graph --bench-file $D/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $D/rem_extract.log 2>&1 || { echo failed:extract; exit 1; }
  touch $IR/.done
fi
cd $CR
rm -rf $D/rem_work_$TAG
python -m modelblaster.pipeline.kb_model_run $IR --cache $D/work/cache --hints $D/work/graph_hints.json --arch 89 \
  --gpu-model "NVIDIA L40S" --reps 5 --work $D/rem_work_$TAG --out $D/rem_$TAG.json --evaluator local --kb $KB \
  --kb-python /tmp/krisd/venv/bin/python --gpus 0 --torch-baseline off > $D/rem_$TAG.log 2>&1
echo "rc=$? $(python3 -c "import json;d=json.load(open('$D/rem_$TAG.json'));print('model_e2e_us', d.get('model_e2e_us'), 'passed', d.get('passed'))" 2>&1 | tail -1)"
nvidia-smi --query-gpu=clocks.mem,clocks.max.mem --format=csv,noheader
