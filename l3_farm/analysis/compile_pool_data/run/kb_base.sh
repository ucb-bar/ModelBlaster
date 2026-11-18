. /tmp/krisd/cpool/env.sh
rm -rf /tmp/krisd/cpool/base; mkdir -p /tmp/krisd/cpool/base; cp -r /tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster /tmp/krisd/cpool/base/KB
cd /tmp/krisd/cpool/base/KB
cp /tmp/krisd/cpool/slots.py.orig src/kernelblaster/knowledge/slots.py; cp /tmp/krisd/cpool/bc_head.py src/kernelblaster/servers/utils/build_cache.py; rm -f src/kernelblaster/knowledge/remote_build.py tests/test_build_pool.py
export PYTHONPATH=$PWD/src
python -m pytest tests/test_fast_paths.py tests/test_loadbalance.py tests/test_knowledge_v2.py -q -p no:cacheprovider -rfE 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | tail
