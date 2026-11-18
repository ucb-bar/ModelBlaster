. /tmp/krisd/venv/bin/activate
CUDA_N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export PATH=$CUDA_N/bin:$PATH CUDA_HOME=$CUDA_N
for L in cmb2 cmb3; do
CR=/tmp/krisd/mc/$L
echo "#### $L"
cd $CR/ModelBlaster && PYTHONPATH=$CR:/tmp/krisd/mc/ip/pylib python -m pytest pipeline/tests -q -p no:cacheprovider -rf > /tmp/krisd/mc/ip/full_$L.mb.log 2>&1; grep -E "^(FAILED|ERROR)|passed|failed" /tmp/krisd/mc/ip/full_$L.mb.log
echo ==KB $L
cd $CR/ModelBlaster/third_party/KernelBlaster && PYTHONPATH=$PWD/src:$PWD/tests:/tmp/krisd/mc/ip/pylib python -m pytest tests -q -p no:cacheprovider -rf > /tmp/krisd/mc/ip/full_$L.kb.log 2>&1; grep -E "^(FAILED|ERROR)|passed|failed" /tmp/krisd/mc/ip/full_$L.kb.log
done
echo DONE
