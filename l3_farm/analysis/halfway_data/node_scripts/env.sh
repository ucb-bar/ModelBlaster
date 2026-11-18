export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
export HOME=/tmp/krisd/hw/home TRITON_CACHE_DIR=/tmp/krisd/hw/home/triton TMPDIR=/tmp/krisd/hw/tmp
mkdir -p $HOME $TMPDIR
. /tmp/krisd/venv/bin/activate
export PYTHONPATH=/tmp/krisd/hw/code
export MB_PRUNE_DEAD=0
