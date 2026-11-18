. /tmp/krisd/cpool/env.sh
KB=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster
export PYTHONPATH=$KB/src:$PYTHONPATH
cd $KB
python -m pytest tests -x -q -p no:cacheprovider --maxfail=1000 -q tests/test_build_pool.py 2>&1 | tail -15
echo ===FULL
timeout 3000 python -m pytest tests -q -p no:cacheprovider --timeout=300 2>&1 | tail -60 || true
