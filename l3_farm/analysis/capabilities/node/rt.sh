. /tmp/krisd/a2/env.sh
export PYTHONPATH=$PYTHONPATH:/tmp/krisd/a2/pyx
cd /tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster && python -m pytest tests/test_a2_capabilities.py -q -p no:cacheprovider -k RealTool 2>&1 | tail -3
