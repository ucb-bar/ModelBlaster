. /tmp/krisd/cpool/env.sh
KB=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster
export PYTHONPATH=$KB/src:/tmp/krisd/cpool/code/ModelBlaster:$PYTHONPATH
cd $KB; python -m pytest -q -p no:cacheprovider tests -k sqlite_concurrent 2>&1 | grep -E "passed|failed|third_party" | head -3
