#!/bin/bash
# Runs on ONE L40S node (one GPU, nothing else of ours running). Layout under $W=/tmp/krisd/abl3cmp: probs/<name>.py, bench.py (E1 BENCH of torch_posthoc.py),
# run_baselines.py, run_baselines_compile.py. Per problem, back to back: (a) run_baselines (1x, iters 100 warmup 10 --unit us), (b) torch_bench x3
# (`bench.py prob name 50 10 0`), (c) run_baselines_compile. Output $W/out/<name>.{rb,tb1,tb2,tb3,rbc}.txt and $W/out/env_*.txt. Detached by the controller.
W=/tmp/krisd/abl3cmp; O=$W/out; mkdir -p $O; cd $W
. /tmp/krisd/venv/bin/activate
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export TRITON_CACHE_DIR=$W/cache/triton TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TMPDIR=$W/cache; mkdir -p $W/cache
snap(){ { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw --format=csv; \
  echo "others:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } >> $O/env_$2.txt; }
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version())" > $O/env_torch.txt 2>&1
hostname >> $O/env_torch.txt; snap start 0
rm -f $W/DONE
for f in $W/probs/*.py; do n=$(basename $f .py); mkdir -p $W/one/$n; ln -sfn $f $W/one/$n/$n.py
  snap before $n
  python run_baselines.py --root $W/one/$n --unit us --iters 100 --warmup 10 --device cuda > $O/$n.rb.txt 2>&1
  for i in 1 2 3; do python bench.py $f $n 50 10 0 2>&1 | grep '@@POSTHOC@@' > $O/$n.tb$i.txt; done
  timeout 900 python run_baselines_compile.py --root $W/one/$n --unit us --iters 100 --warmup 10 --device cuda > $O/$n.rbc.txt 2>&1
  snap after $n
done
snap end 0; touch $W/DONE
