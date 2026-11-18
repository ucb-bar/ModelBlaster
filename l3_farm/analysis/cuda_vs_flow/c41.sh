N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
B=/tmp/krisd/cvf/cache; mkdir -p $B/ext
export TORCH_EXTENSIONS_DIR=$B/ext TMPDIR=$B PATH=/tmp/krisd/baseline_tools/py/bin:$N/bin:/usr/bin:/bin CUDA_HOME=$N TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
. /tmp/krisd/venv/bin/activate
cd /tmp/krisd/cvf/claude41
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
timeout 800 python3 claude41.py sol002.py prob.py 2>&1 | grep -v Warning | tail -20
