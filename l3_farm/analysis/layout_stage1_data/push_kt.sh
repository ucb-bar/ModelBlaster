#!/bin/bash
# push the kernel test files + current library to my node's kt dir: push_kt.sh JOB
cd "$(dirname "$0")"
P=""; for f in kernel_lib.py kernel_test.py bench.py cases_real.json naive_f16.cu; do P="$P --put $f=/tmp/krisd/x142/kt/$f"; done
P="$P --put ../../../ModelBlaster/gpu_eval/harness/cuda/mb_cuda_kernels_f16.cu=/tmp/krisd/x142/kt/mb_cuda_kernels_f16.cu"
python3 nodes_x142.py run $1 ${2:-node_scripts/k1.sh} ${3:-900} $P
