. /tmp/krisd/cpool/env.sh
KB=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster
export PYTHONPATH=$KB/src:$PYTHONPATH
cd $KB
python -m pytest tests -q -p no:cacheprovider -rfE 2>&1 | grep -E "^(FAILED|ERROR|SKIPPED)|passed|failed" | tail -40
