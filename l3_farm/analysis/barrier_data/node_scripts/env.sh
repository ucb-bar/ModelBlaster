export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
export HOME=/tmp/krisd/bar/home TRITON_CACHE_DIR=/tmp/krisd/bar/home/triton TMPDIR=/tmp/krisd/bar/tmp
mkdir -p $HOME $TMPDIR
. /tmp/krisd/venv/bin/activate
export PYTHONPATH=/tmp/krisd/bar/code
export MB_PRUNE_DEAD=0
