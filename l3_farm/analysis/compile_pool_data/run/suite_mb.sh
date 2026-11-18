. /tmp/krisd/cpool/env.sh
KB=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster
export PYTHONPATH=$KB/src:$PYTHONPATH
cd /tmp/krisd/cpool/code/ModelBlaster
python -m pytest pipeline/tests/test_build_cache.py -q -p no:cacheprovider 2>&1 | tail -8
echo ===FULL
timeout 3000 python -m pytest pipeline/tests -q -p no:cacheprovider 2>&1 | tail -60 || true
