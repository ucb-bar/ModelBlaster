ls /tmp/krisd/venv/bin/python 2>&1
C=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
ls $C/bin | grep -i "ncu\|nvcc\|nsight"; ls $C | head -30
ls $C/nsight-compute* 2>&1 | head
ls /usr/local/cuda*/bin 2>&1 | grep -i ncu; which -a ncu nv-nsight-cu-cli
