N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
B=/tmp/krisd/cvf/cache; mkdir -p $B/ext
export TORCH_EXTENSIONS_DIR=$B/ext TMPDIR=$B PATH=/tmp/krisd/baseline_tools/py/bin:$N/bin:/usr/bin:/bin CUDA_HOME=$N TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
. /tmp/krisd/venv/bin/activate
SEC="--clock-control none --section SpeedOfLight --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis"
F='Duration|Compute \(SM\) Throughput|Executed Ipc Active|Grid Size|Registers Per|Achieved Occupancy|highest-utilized|Dynamic Shared|L2 Cache Throughput|DRAM Throughput|SM Busy|^  [a-z_<>:0-9]+.*\(|void'
cd /tmp/krisd/cvf/flow41
for k in k_gru_fused k_gru_tc; do timeout 600 ncu $SEC -k $k --launch-count 1 ./flow_bench short 2>&1 | grep -E "$F" | cut -c1-130; done
cd /tmp/krisd/cvf/claude41
timeout 600 ncu $SEC -k regex:gru_layer --launch-skip 1 --launch-count 1 python3 claude41.py sol002.py prob.py as_submitted 2>&1 | grep -E "$F" | cut -c1-130
