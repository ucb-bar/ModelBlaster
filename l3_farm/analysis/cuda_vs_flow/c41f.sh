N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
B=/tmp/krisd/cvf/cache; mkdir -p $B/ext
export TORCH_EXTENSIONS_DIR=$B/ext TMPDIR=$B PATH=/tmp/krisd/baseline_tools/py/bin:$N/bin:/usr/bin:/bin CUDA_HOME=$N TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
. /tmp/krisd/venv/bin/activate
cd /tmp/krisd/cvf/claude41
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
timeout 600 python3 claude41.py solfinal.py prob.py as_submitted 2>&1 | grep RES
nsys profile -o /tmp/krisd/cvf/c41f --force-overwrite true -t cuda python3 claude41.py solfinal.py prob.py as_submitted >/dev/null 2>&1
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum -f csv /tmp/krisd/cvf/c41f.nsys-rep 2>/dev/null | grep -v "cudnn\|gemmSN\|sgemm\|elemWise" | head -12
