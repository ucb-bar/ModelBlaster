N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
cd /tmp/krisd/cvf/flow41
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
$N/bin/nvcc -O3 -arch=sm_89 -std=c++17 -lineinfo --default-stream per-thread -w flow_bench.cu -o flow_bench 2>&1 | tail -20
timeout 600 ./flow_bench all
echo exit $?
