. /tmp/krisd/a2/env.sh
[ -d /tmp/krisd/a2/pyx/pytest ] || pip -q install --target /tmp/krisd/a2/pyx pytest >/dev/null 2>&1
export PYTHONPATH=$PYTHONPATH:/tmp/krisd/a2/pyx
cd /tmp/krisd/a2/code/cur/ModelBlaster/third_party/KernelBlaster && python -m pytest tests/test_a2_capabilities.py -q -p no:cacheprovider 2>&1 | tail -3
cd /tmp/krisd/a2/code/cur && python -m pytest modelblaster/pipeline/tests/test_region_slots.py modelblaster/pipeline/tests/test_precision_policy.py modelblaster/pipeline/tests/test_graph_prune.py -q -rs -p no:cacheprovider 2>&1 | tail -6
