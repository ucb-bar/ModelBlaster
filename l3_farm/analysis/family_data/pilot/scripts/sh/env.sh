# sourced by every pilot step on the node (same toolchain as l3_farm/remote/worker3.sh; own tree)
export W=/tmp/krisd/family
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
export PYTHONPATH=$W/py MB_TORCH_SRC=$W/src
. /tmp/krisd/farm/relay.env
export KB=$W/ModelBlaster/third_party/KernelBlaster
export MB_KB_ROOT=$KB
