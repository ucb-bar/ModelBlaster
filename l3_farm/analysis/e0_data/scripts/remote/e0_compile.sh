# E0: rerun the torch_posthoc bench (eager / graph / torch.compile) with writable caches under /tmp/krisd/e0/cache
# (the first pass's torch.compile failed: ~/.triton on a full home). Args: <run name> <problem>
R=/tmp/krisd/e0/runs/$1; C=/tmp/krisd/e0/cache; mkdir -p $C/triton $C/inductor
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor TMPDIR=$C
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
. /tmp/krisd/venv/bin/activate
bash -c "exec -a e0_hold_kb_graph_opt_compile sleep 3600" & H=$!
python $R/bench.py $R/problem.py $2 50 10 1 > $R/posthoc2.log 2>&1
grep '^@@POSTHOC@@' $R/posthoc2.log | sed 's/^@@POSTHOC@@//' > $R/posthoc2.json
kill $H
python3 -c "import json;d=json.load(open('$R/posthoc2.json'));print({k:d.get(k) for k in ('eager_us','graph_us','compile_us','compile_error','others_before')})"
