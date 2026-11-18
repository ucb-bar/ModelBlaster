# Node side: quiet re-measurement of an arm's final model + the PyTorch baseline on this node's GPU.
# Args: <work tag (whose final model)> <out label>. Uses the arm's own code root (kb_model_run: whole model from the
# final cache + accepted hints, CUDA-graph replay in the persistent runtime, vs the seed, reps 5) and
# bench.py (torch_posthoc.py BENCH: eager / CUDA graph / torch.compile, median of 50 after 10 warm-up).
TAG=$1; LBL=$2; NAME=43_MinGPTCausalAttention
D=/tmp/krisd/sonnet_$TAG/$NAME; CR=/tmp/krisd/sonnet_code; O=/tmp/krisd/sonnet_remeasure/$LBL
mkdir -p $O
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR
echo "others: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) compute procs; $(pgrep -fc kb_graph_opt) kb_graph_opt"
cd $CR && rm -rf $O/mr && python -m modelblaster.pipeline.kb_model_run $D/irs/$NAME --cache $D/work/cache --hints $D/work/graph_hints.json \
  --arch 89 --reps 5 --work $O/mr --kb $CR/ModelBlaster/third_party/KernelBlaster --kb-python /tmp/krisd/venv/bin/python \
  --torch-baseline off --out $O/model_run.json 2>$O/model_run.err | tail -1
export TRITON_CACHE_DIR=$O/cache/triton TORCHINDUCTOR_CACHE_DIR=$O/cache/inductor
cd $O && python /tmp/krisd/sonnet_scripts/bench.py $D/problem.py $NAME 50 10 1 2>$O/bench.err | grep '^@@POSTHOC@@' | cut -c12- > $O/torch.json
python - $O <<'PY'
import json, sys
o = sys.argv[1]
t = json.load(open(o + "/torch.json"))
m = json.load(open(o + "/model_run.json"))
print("@@RM@@" + json.dumps({"model_e2e_us": m.get("model_e2e_us"), "seed_e2e_us": m.get("seed_e2e_us"), "passed": m.get("passed"),
      "basis": m.get("model_basis"), "jobs": (m.get("timings") or {}), "torch_eager_us": t.get("eager_us"),
      "torch_graph_us": t.get("graph_us"), "torch_compile_us": t.get("compile_us"), "gpu_uuid": t.get("gpu_uuid"),
      "host": t.get("host"), "others": [t.get("others_before"), t.get("others_after")]}))
PY
