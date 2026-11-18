# Args: <stamp> [pytest -k expr]. GPU flow tests (stub LLM, zero spend) on the frozen tree.
A=/tmp/krisd/a4/$1; cd $A
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$A MB_KB_ROOT=$A/ModelBlaster/third_party/KernelBlaster MB_KB_PYTHON=/tmp/krisd/venv/bin/python
export MB_TEST_TMP=/tmp/krisd/a4/tmp MB_TEST_KEEP=1 KB_TRACE=meta
mkdir -p $MB_TEST_TMP
python -m unittest -v ${2:-modelblaster.pipeline.tests.test_graph_search_flow} 2>&1 | tail -80
