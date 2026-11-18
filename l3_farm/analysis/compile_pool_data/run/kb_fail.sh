. /tmp/krisd/cpool/env.sh
KB=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster
export PYTHONPATH=$KB/src:$PYTHONPATH
cd $KB
python -m pytest -q -p no:cacheprovider tests/test_build_pool.py -rs 2>&1 | tail -8
python -m pytest -q -p no:cacheprovider tests/test_fast_paths.py::test_fast_and_cmake_builds_agree tests/test_knowledge_v2.py::test_sqlite_concurrent_writers_union_evidence tests/test_loadbalance.py 2>&1 | grep -E "Error|error|assert|FAILED|passed|failed" | head -30
cd ..; cd /tmp/krisd/cpool/code/ModelBlaster; python -m pytest -q -p no:cacheprovider pipeline/tests/test_build_cache.py 2>&1 | tail -3
