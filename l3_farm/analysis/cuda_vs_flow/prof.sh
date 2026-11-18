N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
B=/tmp/krisd/cvf/cache; mkdir -p $B/ext
export TORCH_EXTENSIONS_DIR=$B/ext TMPDIR=$B PATH=/tmp/krisd/baseline_tools/py/bin:$N/bin:/usr/bin:/bin CUDA_HOME=$N TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
. /tmp/krisd/venv/bin/activate
cd /tmp/krisd/cvf/claude41
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis"
nsys profile -o /tmp/krisd/cvf/c41 --force-overwrite true -t cuda python3 claude41.py sol002.py prob.py as_submitted >/dev/null 2>&1
nsys stats -r cuda_gpu_kern_sum -f csv /tmp/krisd/cvf/c41.nsys-rep 2>/dev/null | head -8
timeout 600 ncu $SEC -k regex:gru_layer --launch-skip 1 --launch-count 1 python3 claude41.py sol002.py prob.py as_submitted 2>&1 | grep -v "^==PROF==\|Warning" | sed -n '/gru_layer/,$p' | head -120
cd /tmp/krisd/cvf/flow41
timeout 600 ncu $SEC -k regex:"v0::k_gru_fused" --launch-count 1 ./flow_bench short 2>&1 | sed -n '/k_gru_fused/,$p' | head -110
timeout 600 ncu $SEC -k regex:"v1::k_gru_tc" --launch-count 1 ./flow_bench short 2>&1 | sed -n '/k_gru_tc/,$p' | head -110
