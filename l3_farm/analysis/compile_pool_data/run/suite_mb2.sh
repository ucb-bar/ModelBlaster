. /tmp/krisd/cpool/env.sh
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:$PYTHONPATH
cd /tmp/krisd/cpool/code/ModelBlaster
python -m pytest pipeline/tests/test_build_cache.py -q -p no:cacheprovider 2>&1 | tail -3
echo ===FULL
python -m pytest pipeline/tests -q -p no:cacheprovider -rf 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | tail -60
